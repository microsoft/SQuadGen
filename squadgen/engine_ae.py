# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import os
import json
import time
from typing import Iterable

import torch

import squadgen.util.misc as misc
import squadgen.util.lr_sched as lr_sched

from squadgen.network.models_vae_joint import KLAutoEncoderJoint
from squadgen.network.utils import *

def calc_loss(batch, model, model_without_ddp: KLAutoEncoderJoint, split, args=None, epoch_real_float=None):

    label = model_without_ddp.get_label(batch)
    
    out = model.forward(batch, is_cond_mode=args.use_geom_cond_mean)
    
    log_all_b = {}
    log = {}
    log_part = {}
    loss_total = 0
    
    # recon loss
    if model_without_ddp.freeze_cond_vae:
        loop_list = [[out['out']['out'], label['gt'], "color", model_without_ddp.latent_vae]]
    else:
        loop_list = [[out['out']['out'], label['gt'], "color", model_without_ddp.latent_vae],
            [out['out']['out_cond'], label['gt_cond'], "geom", model_without_ddp.cond_vae]]
    for (pred, gt, name, vae) in loop_list:
        if vae is None:
            continue
        for k in pred.keys():

            x = torch.abs(pred[k] - gt[k])
            x_b = x.mean(dim=[1, 2]) # [B]
            
            assert x_b.shape[0] == batch["batch_size"]
            assert len(x_b.shape) == 1

            log_all_b[f"{split}/{name}_rec_loss_{k}"] = x_b.detach()
            log[f"{split}/{name}_rec_loss_{k}"] = x_b.detach().mean()
            log_part[f"{name}_{k}"] = x_b.detach().mean()

            weight = {vae.out_keys_list[idx]: vae.out_weights_list[idx] for idx in range(len(vae.out_keys_list))}
            loss_total += x_b.mean() * weight[k]

    # kl loss
    if model_without_ddp.freeze_cond_vae:
        loop_list = [[out["latent"]['kl'], "color", model_without_ddp.latent_vae_params.loss_params.kl_weight, model_without_ddp.latent_vae]]
    else:
        loop_list = [[out["latent"]['kl'], "color", model_without_ddp.latent_vae_params.loss_params.kl_weight, model_without_ddp.latent_vae],
                        [out["latent"]['kl_cond'], "geom", model_without_ddp.cond_vae_params.loss_params.kl_weight, model_without_ddp.cond_vae]]
    for (kl, name, kl_weight, vae) in loop_list:
        if vae is None:
            continue
        if model_without_ddp.freeze_cond_vae and name == "geom": # only train color
            continue
        kl_weight_real = kl_weight
        kl_loss = kl.mean() * kl_weight_real
        log[f"{split}/{name}_kl_loss"] = kl_loss.detach()
        loss_total += kl_loss

    log_other = {}
    return {
        "loss_total": loss_total,
        "log": log,
        "log_part": log_part,
        "log_per_recon": log_all_b,
        "log_other": log_other,
        "out": out,
        "label": label,
    }


def train_one_epoch(model: torch.nn.Module, model_without_ddp: KLAutoEncoderJoint,
                    data_loader: Iterable, dataset_train, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None):
    time_train_start = time.time()
    model.train(True)
    if model_without_ddp.freeze_cond_vae:
        model_without_ddp.cond_vae.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    num_gpu = misc.get_world_size()
    time_cur = time.time()

    mix_precision_dtype = None
    if args.mix_precision_dtype=="fp16":
        mix_precision_dtype = torch.float16
    elif args.mix_precision_dtype=="bf16":
        mix_precision_dtype = torch.bfloat16
    elif args.mix_precision_dtype=="fp32":
        mix_precision_dtype = torch.float32
    else:
        raise NotImplementedError

    end_time = time.time()
    time_data = 0

    dataset_train.epoch = epoch

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        time_data += time.time() - end_time

        epoch_real_float = (data_iter_step / len(data_loader) * args.training_data_repeat_times + epoch)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, epoch_real_float, args)

        batch = to(batch, device=device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=bool(args.mix_precision), dtype=mix_precision_dtype):
            loss_dict = calc_loss(batch, model, model_without_ddp, split="train", args=args, epoch_real_float=epoch_real_float)

        loss = loss_dict["loss_total"]
        loss /= accum_iter
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        metric_logger.update(loss=loss*accum_iter)

        for k, v in loss_dict["log_part"].items():
            if k not in metric_logger.meters:
                metric_logger.add_meter(k, misc.SmoothedValue(window_size=20, fmt='{value:.3f}'))
            metric_logger.update(**{k: v})

        for k, v in loss_dict["log_other"].items():
            if k not in metric_logger.meters:
                metric_logger.add_meter(k, misc.SmoothedValue(window_size=20, fmt='{value:.3f}'))
            metric_logger.update(**{k: v})

        for i in range(len(optimizer.param_groups)):
            lr = optimizer.param_groups[i]["lr"]
            k, v = f'lr_{i}', lr
            if k not in metric_logger.meters:
                metric_logger.add_meter(k, misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.update(**{k: v})

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            # """ We use the number of data the model has seen as the x-axis.
            # This calibrates different curves when batch size changes.
            # """
            step = int(epoch_real_float * 1000) # epoch_1000x
            # step = data_iter_step * batch["batch_size"] + epoch * len(data_loader) * batch["batch_size"] # num_data
            
            # log loss
            log_writer.add_scalar('loss', loss*accum_iter, step)
            
            # log lr
            for i in range(len(optimizer.param_groups)):
                log_writer.add_scalar(f'lr_{i}', optimizer.param_groups[i]["lr"], step)

            # log loss, part
            log_writer.log(loss_dict["log"], step)

            # log real epoch
            log_writer.add_scalar(f'epoch_real', int(epoch_real_float), step)

            # log time per data
            time_cost = time.time() - time_cur
            log_writer.add_scalar(f'time_per_data(ms)', time_cost * 1000 / (accum_iter * num_gpu * batch["batch_size"]), step)
            time_cur = time.time()

            log_writer.add_scalar(f'grad_norm', grad_norm, step)

        end_time = time.time()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if log_writer is not None:
        log_writer.add_scalar("time_train_epoch(min)", (time.time() - time_train_start)/60, (epoch+args.training_data_repeat_times)*1000)
        log_writer.add_scalar("time_train_dataloader(min)", (time_data)/60, (epoch+args.training_data_repeat_times)*1000)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, dataset_val, model, model_without_ddp, device, results_dir, log_writer, epoch, args):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    res = 256

    time_eval_start = time.time()

    model.eval()

    val_results = {}

    mix_precision_dtype = None
    if args.mix_precision_dtype=="fp16":
        mix_precision_dtype = torch.float16
    elif args.mix_precision_dtype=="bf16":
        mix_precision_dtype = torch.bfloat16
    elif args.mix_precision_dtype=="fp32":
        mix_precision_dtype = torch.float32
    else:
        raise NotImplementedError

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, 16, header)):

        batch = to(batch, device=device, non_blocking=True)

        data_type_list = batch["data_type_list"]
        data_index_list = batch["index_list"]
        batch_size = batch["batch_size"]
        prefix = f"val_{data_type_list[0]}"
        for data_type in data_type_list:
            assert data_type == data_type_list[0]
        data_type = data_type_list[0]

        # inference
        with torch.cuda.amp.autocast(enabled=bool(args.mix_precision), dtype=mix_precision_dtype):
            loss_dict = calc_loss(batch, model, model_without_ddp, split=prefix, args=args, epoch_real_float=epoch)

        ans_recon = {
            **loss_dict["out"]["out"]["out"],
            **loss_dict["out"]["out"]["out_cond"],
        }
        ans_gt = {
            **loss_dict["label"]["gt"],
            **loss_dict["label"]["gt_cond"],
        }

        for k in ans_recon.keys():
            ans_recon[k] = ans_recon[k].to(torch.float32)
        for k in ans_gt.keys():
            ans_gt[k] = ans_gt[k].to(torch.float32)

        img_list_dict = {}
        for index_global, i in zip(data_index_list, range(batch_size)):
            name = dataset_val.filelist[index_global].replace("/", "_")
            save_path = os.path.join(results_dir, f"{index_global}_{data_type}_{name}")
            os.makedirs(save_path, exist_ok=True)

            if "color" in ans_recon:
                gt_posterior = loss_dict["out"]["latent"]["lat_enc"]["posterior"]
            else:
                gt_posterior = loss_dict["out"]["latent"]["cond_enc"]["posterior"]
            
            latent_feature_mean_all = gt_posterior.mean

            is_show_last_tsne = False
            if is_show_last_tsne and "color" not in ans_recon:
                with torch.cuda.amp.autocast(enabled=bool(args.mix_precision), dtype=mix_precision_dtype):
                    latent_feature_mean_all = model_without_ddp.cond_vae.return_last_features_and_norm(latent_feature_mean_all)
    
            vae_mean = latent_feature_mean_all[i]
            torch.save(vae_mean.cpu(), os.path.join(save_path, "vae_mean.pth"))

            # save original data
            data_path = dataset_val.filelist[index_global]
            if not data_path.startswith("/"):
                data_path = os.path.join(dataset_val.folder, data_path)
            os.system(f"cp {data_path} {save_path}/data_original.{data_path.split('.')[-1]}")

            # calc scores and log scores
            tmp = {"loss": {}, "info": {}}
            score = {}
            for k in ans_recon.keys():
                t = calc_score_mse(ans_recon[k][i], ans_gt[k][i])
                score[k] = t
                for kk in t:
                    k_tmp, v_tmp = f"{prefix}/{k}_{kk}", t[kk]
                    tmp["info"][k_tmp] = v_tmp
                    metric_logger.update(**{k_tmp: v_tmp})
            for k in loss_dict["log_per_recon"].keys():
                k_tmp, v_tmp = f"{k}", loss_dict["log_per_recon"][k][i].item()
                tmp["loss"][k_tmp] = v_tmp
                metric_logger.update(**{k_tmp: v_tmp})
            
            # record std
            tmp["info"]["std_mean"] = gt_posterior.std.mean().item()
            metric_logger.update(**{f"{prefix}/std_mean": gt_posterior.std.mean().item()})

            with open(os.path.join(save_path, "score.json"), "w") as f:
                json.dump(tmp, f, indent=4)

            # save images
            for k in ans_recon.keys():
                if args.is_render_image == 0: continue
                
                if k.endswith("_near") or k.endswith("_global"):
                    xyz = batch[f"xyz_{k.split('_')[-1]}"][i]
                elif k.endswith("_edge"):
                    xyz = batch["xyz_queryedge"][i]
                    continue
                else:
                    xyz = batch["xyz_query"][i]
                    
                if k in ["dcdf", "dcdf_edge", "cdf", "cdf_edge"]:
                    pred = ans_recon[k][i]
                    gt = ans_gt[k][i]
                    img = save_and_return_image_gcolor(xyz, pred, gt,
                                          save_path=save_path, name_suffix=f"_{k}_raw", res=res)
                    img_list_dict[f"img_{k}_raw"] = img_list_dict.get(f"img_{k}_raw", []) + [{
                        "image": img,
                        "caption": f'MSE: {score[k]["mse"]}, L1: {score[k]["L1"]}',
                    }]
                elif k.startswith("udf"):
                    img = save_and_return_image_udf(xyz, ans_recon[k][i], ans_gt[k][i], 
                                                save_path=save_path, name_suffix=f"_{k}", res=res)
                    img_list_dict[f"img_{k}"] = img_list_dict.get(f"img_{k}", []) + [{
                        "image": img,
                        "caption": f'MSE: {score[k]["mse"]}.',
                    }]
                elif k.startswith("offset") or k.startswith("doffset") or k.startswith("foffset"):
                    continue
                else:
                    raise NotImplementedError    
            img_dict = {}
            for k in img_list_dict:
                caption = img_list_dict[k][-1]["caption"]
                caption = f"GT, preds. {dataset_val.filelist[index_global]}. {caption}"
                img_dict[k] = {
                    "image": img_list_dict[k][-1]["image"],
                    "caption": caption,
                }
            val_results[data_type] = val_results.get(data_type, []) + [{
                "index_global": index_global,
                "img_dict": img_dict,
                "score": score,
                "save_path": save_path,
            }]

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # log metrics
    for k in metric_logger.meters:
        if log_writer is not None:
            log_writer.add_scalar(k, metric_logger.meters[k].global_avg, epoch*1000)

    outputs = val_results
    
    for data_type in outputs:
        N = 8
        wandb_img_list_dict = {}
        for i in range(len(outputs[data_type])):
            info = outputs[data_type][i]["img_dict"]
            save_path = outputs[data_type][i]["save_path"]
            index_global = outputs[data_type][i]["index_global"]
            
            # log image
            if i < N:
                for k in info:
                    if k.startswith("img_"):
                        img = info[k]["image"]
                        caption = info[k]["caption"]
                        wandb_img_list_dict[k] = wandb_img_list_dict.get(k, []) + [{
                            "img": img,
                            "caption": caption,
                        }]
            
        for k in wandb_img_list_dict:
            wandb_img_list = wandb_img_list_dict[k]
            if len(wandb_img_list) > 0 and log_writer is not None:
                log_writer.add_image_list(f"eval_{data_type}/{k}", wandb_img_list, epoch*1000)

    if log_writer is not None:
        log_writer.add_scalar("time_eval_all(min)", (time.time() - time_eval_start)/60, epoch*1000)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}