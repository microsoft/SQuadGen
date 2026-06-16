# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import json
import time
from typing import Iterable

import torch

import squadgen.util.misc as misc
import squadgen.util.lr_sched as lr_sched

from squadgen.network.models_sit import SiT, StackedRandomGenerator
from squadgen.network.utils import *

def train_one_epoch(model: torch.nn.Module, ae: torch.nn.Module, model_without_ddp: SiT, criterion: torch.nn.Module,
                    data_loader: Iterable, dataset_train, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None):
    time_train_start = time.time()
    model.train(True)
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

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        time_data += time.time() - end_time

        epoch_real_float = (data_iter_step / len(data_loader) * args.training_data_repeat_times + epoch)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, epoch_real_float, args)

        batch = to(batch, device=device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=bool(args.mix_precision), dtype=mix_precision_dtype):
            # drop condition prob = 0.1, follow the dalle-2 paper: https://arxiv.org/pdf/2204.06125, or does not drop condition
            is_drop_cond=torch.rand(1)<args.drop_prob
            
            with torch.no_grad():
                tmp = model_without_ddp.get_latent_and_condition(batch, ae, is_drop_cond=is_drop_cond, is_mode=args.is_mode)
            x = tmp["latent"]
            condition_params = tmp["condition_params"]
            latent_dict = tmp["latent_dict"]

            loss_dict = criterion(model, x, condition_params=condition_params,
                            args=args,
                            ae=ae, batch=batch, latent_dict=latent_dict)
            loss = 0
            for k in loss_dict:
                metric_logger.update(**{k: loss_dict[k].item()})
                loss += loss_dict[k]

        loss /= accum_iter
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        metric_logger.update(loss=loss)
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
            step = int(epoch_real_float * 1000) # epoch_1000x

            log_writer.add_scalar('loss', loss*accum_iter, step)
            # log lr
            for i in range(len(optimizer.param_groups)):
                log_writer.add_scalar(f'lr_{i}', optimizer.param_groups[i]["lr"], step)

            # log loss
            for k in loss_dict:
                log_writer.add_scalar(f"train/{k}", loss_dict[k].item()*accum_iter, step)

            # log time per data
            time_cost = time.time() - time_cur
            log_writer.add_scalar(f'time_per_data(ms)', time_cost * 1000 / (accum_iter * num_gpu * batch["batch_size"]), step)
            time_cur = time.time()

            log_writer.add_scalar(f'grad_norm', grad_norm, step)

        end_time = time.time()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if log_writer is not None:
        log_writer.add_scalar("time_train_epoch(min)", (time.time() - time_train_start)/60, (epoch+args.training_data_repeat_times)*1000)
        log_writer.add_scalar("time_train_dataloader(min)", (time_data)/60, (epoch+args.training_data_repeat_times)*1000)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def sample(model, model_without_ddp: SiT, condition_params, device, batch_size, seeds=0, args=None, criterion=None):
    batch_seeds = torch.arange(batch_size*seeds, batch_size*(seeds+1), device=device)

    rnd = StackedRandomGenerator(device, batch_seeds)
    latents = rnd.randn([batch_size, model_without_ddp.n_latents, model_without_ddp.channels], device=device)

    transport_sampler = args.sit_transport_sampler
    sample_fn = transport_sampler.sample_ode()
    model_fn = model_without_ddp.forward
    sample_model_kwargs = {
        "condition_params": condition_params,
    }
    samples = sample_fn(latents, model_fn, **sample_model_kwargs)[-1]
    return model_without_ddp.denormalize_latents(samples)


def evaluate(data_loader, dataset_val, model, model_without_ddp: SiT, ae, criterion, device, results_dir, log_writer, epoch, args):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    res = 256
    time_eval_start = time.time()

    # switch to evaluation mode
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
        batch_size = batch["batch_size"]
        data_index_list = batch["index_list"]
        data_type_list = batch["data_type_list"]
        prefix = f"val_{data_type_list[0]}"
        for data_type in data_type_list:
            assert data_type == data_type_list[0]
        data_type = data_type_list[0]

        with torch.cuda.amp.autocast(enabled=bool(args.mix_precision), dtype=mix_precision_dtype):
            with torch.no_grad():
                tmp = model_without_ddp.get_latent_and_condition(batch, ae, is_mode=args.is_mode)

                x = tmp["latent"]
                condition_params = tmp["condition_params"]
                latent_dict = tmp["latent_dict"]
                lat_gt = x

                ans_gt = ae.latent_vae.decode_batch(latent_dict["lat_end"], batch)
                gt_posterior = latent_dict["lat_enc"]["posterior"]

                ans_gt["latent"] = lat_gt

        if True:
            for index_global, i in zip(data_index_list, range(batch_size)):
                name = dataset_val.filelist[index_global].replace("/", "_")
                save_path = os.path.join(results_dir, f"{index_global}_{data_type}_{name}")
                os.makedirs(save_path, exist_ok=True)

                # save x
                torch.save(x[i].to(torch.float32), os.path.join(save_path, f"latent_target.pt"))

                with open(os.path.join(save_path, f"input_info.json"), 'w') as f:
                    gt_info = {
                        "x": {
                            "shape": x.shape,
                            "max": x.max().item(),
                            "min": x.min().item(),
                            "mean": x.mean().item(),
                            "std": x.std().item(),
                            "abstract": x.reshape(-1)[:5].to(torch.float32).cpu().numpy().tolist(),
                        },
                        "condition_global": {
                            "shape": [x.shape for x in condition_params["condition_global"]],
                            "mean": [x.mean().item() for x in condition_params["condition_global"]],
                            "abstract": [x.reshape(-1)[:5].to(torch.float32).cpu().numpy().tolist() for x in condition_params["condition_global"]],
                        },
                        "file": dataset_val.filelist[batch["index_list"][0]],
                    }
                    if condition_params["condition"] is not None:
                        gt_info["condition"] = {
                            "shape": condition_params["condition"].shape,
                            "max": condition_params["condition"].max().item(),
                            "min": condition_params["condition"].min().item(),
                            "mean": condition_params["condition"].mean().item(),
                            "std": condition_params["condition"].std().item(),
                            "abstract": condition_params["condition"].reshape(-1)[:5].to(torch.float32).cpu().numpy().tolist(),
                        }
                    json.dump(gt_info, f, indent=4)

        n_gen = args.n_gen
        results_all = []
        for gen_idx in range(n_gen):

            with torch.cuda.amp.autocast(enabled=bool(args.mix_precision), dtype=mix_precision_dtype):
                with torch.no_grad():

                    seeds = gen_idx

                    time_start = time.time()
                    lat = sample(model, model_without_ddp, condition_params, device, batch_size=batch["xyz"].shape[0], seeds=seeds, args=args, criterion=criterion)
                    lat = lat.to(torch.float32)
                    print(f"sample time: {time.time() - time_start:.2f}s for batch size {batch['xyz'].shape[0]}")

                    latent_dict["lat"] = lat
                    latent_dict["lat_end"] = latent_dict["lat"]
                    out = ae.latent_vae.decode_batch(latent_dict["lat_end"], batch)
            
            ans_recon = out
            ans_recon["latent"] = lat

            img_list_dict = {}
            scores = []

            # save to file
            for index_global, i in zip(data_index_list, range(batch_size)):
                name = dataset_val.filelist[index_global].replace("/", "_")
                save_path = os.path.join(results_dir, f"{index_global}_{data_type}_{name}")
                os.makedirs(save_path, exist_ok=True)

                if gen_idx == 0:
                    # save gt info
                    with open(os.path.join(save_path, f"gt_info.json"), 'w') as f:
                        vae_latent_mean = gt_posterior.mean[i]
                        vae_latent_std = gt_posterior.std[i]
                        gt_info = {
                            "vae_latent_std_L1": vae_latent_std.abs().mean().item(),
                            "vae_latent_std_mse": (vae_latent_std**2).mean().item(),
                        }
                        json.dump(gt_info, f, indent=4)

                 # calc scores and log scores
                tmp = {"info": {}}
                score = {}
                for k in ans_gt.keys():
                    t = calc_score_mse(ans_recon[k][i], ans_gt[k][i])
                    score[k] = t
                    for kk in t:
                        k_tmp, v_tmp = f"{prefix}/{k}_{kk}", t[kk]
                        tmp["info"][k_tmp] = v_tmp
                scores.append(tmp["info"])
                with open(os.path.join(save_path, f"score_{gen_idx}.json"), "w") as f:
                    json.dump(tmp, f, indent=4)

                # save images
                for k in ans_recon.keys():     
                    if args.is_render_image == 0:
                        is_net = False
                        is_gt = False
                        is_err = False
                    else:
                        is_gt = gen_idx == 0
                        is_net = True
                        is_err = True
                        
                    if k in ["latent", "udf_near", "udf"]:
                        continue               
                    if k.endswith("_edge"):
                        continue
                    if k.endswith("_near") or k.endswith("_global"):
                        xyz = batch[f"xyz_{k.split('_')[-1]}"][i]
                    else:
                        xyz = batch["xyz_query"][i]

                    if k.startswith("offset") or k.startswith("foffset") or k.startswith("doffset"):
                        continue
                    elif k in ["dcdf", "dcdf_edge", "cdf", "cdf_edge"]:
                        if k not in ["dcdf"]:
                            continue
                        pred_color = convert_gcolor_to_gcolormap(ans_recon[k][i])
                        gt_color = convert_gcolor_to_gcolormap(ans_gt[k][i])
                        ans = save_and_return_image_split(xyz, pred_color, gt_color,
                                                save_path=save_path, name_suffix=f"_{k}_{gen_idx}", res=res, data_type="color4", is_gt=is_gt, is_net=is_net, is_err=is_err)
                        img_list_dict[f"img_{k}_raw"] = img_list_dict.get(f"img_{k}_raw", []) + [{
                            **ans,
                            "caption": f'MSE: {score[k]["mse"]}, L1: {score[k]["L1"]}',
                        }]

                    else:
                        raise NotImplementedError
            results_all.append({
                "score": scores,
                "image_dict": img_list_dict
            })

        # collect multi generation results
        for index_global, i in zip(data_index_list, range(batch_size)):
            name = dataset_val.filelist[index_global].replace("/", "_")
            save_path = os.path.join(results_dir, f"{index_global}_{data_type}_{name}")
            os.makedirs(save_path, exist_ok=True)

            # log score
            tmp = dict()
            for gen_idx in range(n_gen):
                for k in results_all[gen_idx]["score"][i]:
                    tmp[k] = tmp.get(k, []) + [results_all[gen_idx]["score"][i][k]]
            for k in tmp:
                metric_logger.update(**{k: get_optimal(k, tmp[k])})

            # log image
            tmp = dict()
            for gen_idx in range(n_gen):
                for k in results_all[gen_idx]["image_dict"]:
                    x = results_all[gen_idx]["image_dict"][k][i]
                    if k not in tmp:
                        tmp[k] = {
                            "image": [x.get("image_gt_raw", None), x.get("img_gt", None)],
                            "caption": f"GT, preds. {dataset_val.filelist[index_global]}."                            
                        }
                    tmp[k]["image"].append(x.get("img", None))
                    tmp[k]["image"].append(x.get("img_err", None))

                    for k_score in ["dcdf_L1"]:
                        for k_score_ in results_all[gen_idx]["score"][i]:
                            if k_score_.endswith(k_score):
                                k_cap = f"caption_{k_score}"
                                if k_cap not in tmp[k]:
                                    tmp[k][k_cap] = f"{k_score}: "
                                tmp[k][k_cap] += f"{results_all[gen_idx]['score'][i][k_score_]:.4f}, "
                                break

            for k in tmp:
                tmp[k]["image"] = [x for x in tmp[k]["image"] if x is not None]
                if len(tmp[k]["image"]) == 0:
                    continue
                img = np.concatenate(tmp[k]["image"], axis=1)
                cv2.imwrite(os.path.join(save_path, f"{k}.png"), cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
                tmp[k]["image"] = img
                for k_cap in tmp[k]:
                    if k_cap.startswith("caption_"):
                        tmp[k]["caption"] += tmp[k][k_cap]
            val_results[data_type] = val_results.get(data_type, []) + [(index_global, tmp)]

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
            index_global = outputs[data_type][i][0]
            info = outputs[data_type][i][1]

            # log image
            if i < N:
                for k in info:
                    if k.startswith("img_"):
                        img = info[k]["image"]
                        if len(img) == 0:
                            continue
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
