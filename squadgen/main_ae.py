import argparse
import datetime
import copy
import numpy as np
import os
import time
from pathlib import Path
from omegaconf import OmegaConf
import torch
import torch.backends.cudnn as cudnn

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import squadgen.util.misc as misc
from squadgen.util.datasets import get_dataloader
from squadgen.util.misc import NativeScalerWithGradNormCount as NativeScaler
from squadgen.util.logger import Logger

import squadgen.network.models_vae_joint as models_vae_joint
from squadgen.network.utils import copy_code, count_parameters

from squadgen.engine_ae import train_one_epoch, evaluate
import torch.distributed as dist

def sync_ddp_hook(state, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    """
    DDP communication hook.
    """
    group_to_use = dist.group.WORLD
    world_size = group_to_use.size()
    grad = bucket.buffer()
    grad.div_(world_size)
    dist.all_reduce(grad, group=group_to_use)
    fut = torch.futures.Future()
    fut.set_result(grad)
    return fut

def get_args_parser():
    parser = argparse.ArgumentParser('Autoencoder', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--debug', default=0, type=int, help='debug mode')
    parser.add_argument('--report_to', default="wandb", type=str)
    parser.add_argument('--save_model_per_epoch', default=1, type=int)
    parser.add_argument('--val_model_per_epoch', default=1, type=int)
    parser.add_argument('--continue_training', default=1, type=int)
    parser.add_argument('--n_sample_surface_query', default=2048, type=int, help="number of query points for training")
    parser.add_argument('--n_sample_near', default=1024, type=int, help="number of near points for geom ae training")
    parser.add_argument('--n_sample_global', default=1024, type=int, help="number of global points for geom ae training")
    parser.add_argument('--n_sample_surface_query_val', default=60000, type=int, help="number of query points for validation")
    parser.add_argument('--n_sample_near_val', default=4096, type=int, help="number of near points for geom ae validation")
    parser.add_argument('--n_sample_global_val', default=4096, type=int, help="number of global points for geom ae validation")
    parser.add_argument('--training_determistic', default=0, type=int)
    parser.add_argument('--pretrained_geom_vae_pth', default="", type=str)
    parser.add_argument('--use_geom_cond_mean', default=0, type=int)
    parser.add_argument('--labeling_fn', default="", type=str)
    parser.add_argument('--training_epoch_len', default=450000, type=int)
    parser.add_argument('--dist_hook', default=0, type=int)
    parser.add_argument('--is_render_image', default=1, type=int)
    parser.add_argument('--val_dataset_type_list', default="test,train", type=str)
    parser.add_argument('--is_skip_first_val', default=0, type=int)
    parser.add_argument('--res', default=-1, type=int, help="number of latent tokens, if > 0, will override the config file")

    # Model parameters
    parser.add_argument('--model_config', required=True, type=str, metavar='MODEL',
                        help='Name of model to train')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=float, default=40, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--mix_precision', type=int, default=1)
    parser.add_argument('--mix_precision_dtype', type=str, default="fp16")


    # Dataset parameters
    parser.add_argument('--dataset_folder', type=str, required=True,
                        help='dataset path')
    parser.add_argument('--data_filter_name', type=str, default="",
                        help='data filter name')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--val_num', default=32, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume_from_another', default='', help='resume from checkpoint')
    parser.add_argument('--training_data_repeat_times', default=1, type=int, 
                        help='repeat times for training data')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--num_workers', default=60, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = Logger(method=args.report_to, log_dir=args.log_dir, name=args.name, resume=args.resume)
    else:
        log_writer = None

    model_config = OmegaConf.load(args.model_config)
    if args.res > 0:
        model_config.model.params.latent_vae_params.num_latents = args.res
        model_config.model.params.cond_vae_params.num_latents = args.res
    model = models_vae_joint.create_joint_vae_from_config(model_config.model.params)
    model.to(device)

    def get_fps_num(config):
        is_learnable_latents = config.is_learnable_latents if hasattr(config, "is_learnable_latents") else 0
        if is_learnable_latents:
            return 0
        return config.num_latents

    num_latents_list = [get_fps_num(model_config.model.params.cond_vae_params)]
    if args.pretrained_geom_vae_pth != "":
        assert model.latent_vae is not None
        checkpoint = misc.load_model_chunk(args.pretrained_geom_vae_pth)
        pretrained_vae = checkpoint['model']
        pretrained_vae = {k.replace("cond_vae.", ""): v for k, v in pretrained_vae.items() if k.startswith("cond_vae.")}
        model.cond_vae.load_state_dict(pretrained_vae)
        model.cond_vae.requires_grad_(False)
        model.cond_vae.eval()
        model.freeze_cond_vae = True
        num_latents_list.append(get_fps_num(model_config.model.params.latent_vae_params))

        # load color model if resume exists
        model_copy = copy.deepcopy(model)
        misc.load_model(args=args, model_without_ddp=model_copy)
        model.latent_vae = copy.deepcopy(model_copy.latent_vae)
    else:
        args.use_geom_cond_mean = 0
        misc.load_model(args=args, model_without_ddp=model)

    model_without_ddp = model

    print("Model = %s" % str(model_without_ddp))
    print(f'number of params (M): {count_parameters(model_without_ddp)}')

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size

    print("base lr: %.2e" % (args.blr))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("batch size: %d" % args.batch_size)
    print("effective batch size: %d" % eff_batch_size)
    print("num_tasks: %d" % num_tasks)
    print("global_rank: %d" % global_rank)

    dataset_dict = get_dataloader(args, num_latents_list=num_latents_list)
    data_loader_val = dataset_dict["val_loader"]
    data_loader_train = dataset_dict["train_loader"]
    dataset_val = dataset_dict["val"]
    dataset_train = dataset_dict["train"]

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        if args.dist_hook:
            model.register_comm_hook(None, sync_ddp_hook)
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr)
    loss_scaler = NativeScaler()
    
    if args.clip_grad is not None and args.clip_grad < 0:
        args.clip_grad = None

    if not args.is_skip_first_val:
        evaluate(data_loader_val, dataset_val, model, model_without_ddp, device,
                results_dir=os.path.join(args.output_dir, 'eval', f"epoch_{args.start_epoch}"),
                log_writer=log_writer, epoch=args.start_epoch, args=args,
                )

    if args.eval:
        exit(0)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs, args.training_data_repeat_times):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, model_without_ddp, data_loader_train, dataset_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            args=args,
        )
        epoch_nxt = epoch + args.training_data_repeat_times
        if args.output_dir and (epoch % args.save_model_per_epoch == 0 or epoch_nxt == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch_nxt)

        if epoch % args.val_model_per_epoch == 0 or epoch_nxt == args.epochs:
            evaluate(data_loader_val, dataset_val, model, model_without_ddp, device,
                        results_dir=os.path.join(args.output_dir, 'eval', f"epoch_{epoch_nxt}"),
                        log_writer=log_writer, epoch=epoch_nxt, args=args,
                        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.name)

    misc.init_distributed_mode(args)
    
    if args.resume_from_another != "" and misc.get_rank() == 0:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        if not os.path.exists(args.resume_from_another):
            import shutil
            print(f"Load resume_from_another, chunk mode")

            # chunk mode
            chunk_idx = 0
            while True:
                if not os.path.exists(f"{args.resume_from_another}.chunk{chunk_idx}"):
                    break
                tgt = os.path.join(args.output_dir, os.path.basename(args.resume_from_another) + f".chunk{chunk_idx}")
                if not os.path.exists(tgt):
                    print(f"copying {args.resume_from_another}.chunk{chunk_idx} to {tgt}")
                    shutil.copyfile(f"{args.resume_from_another}.chunk{chunk_idx}", tgt)
                else:
                    print(f"skip copying {args.resume_from_another}.chunk{chunk_idx} to {tgt}")
                chunk_idx += 1

        else:
            print(f"Load resume_from_another")
            tgt = os.path.join(args.output_dir, os.path.basename(args.resume_from_another))
            if not os.path.exists(tgt):
                import shutil
                shutil.copyfile(args.resume_from_another, tgt)
    torch.distributed.barrier()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(args.log_dir, exist_ok=True)

    if args.epochs % args.training_data_repeat_times != 0:
        args.epochs = args.epochs + args.training_data_repeat_times - args.epochs % args.training_data_repeat_times 
    if args.save_model_per_epoch % args.training_data_repeat_times != 0:
        args.save_model_per_epoch = args.save_model_per_epoch + args.training_data_repeat_times - args.save_model_per_epoch % args.training_data_repeat_times
    if args.val_model_per_epoch % args.training_data_repeat_times != 0:
        args.val_model_per_epoch = args.val_model_per_epoch + args.training_data_repeat_times - args.val_model_per_epoch % args.training_data_repeat_times

    if args.num_workers == -1:
        args.num_workers = os.cpu_count()

    if not args.debug and misc.get_rank() == 0:
        copy_code(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), args.output_dir)

    if args.continue_training:
        # all checkpoint-*.pth are considered as checkpoints
        ckpts = [ckpt for ckpt in os.listdir(args.output_dir) if ckpt.startswith("checkpoint-") and (ckpt.endswith(".pth") or ckpt.endswith(".chunk0"))]
        max_epoch = -1
        for ckpt in ckpts:
            epoch = ckpt.split("-")[1].split(".")[0]
            try:
                int(epoch)
            except:
                continue
            if int(epoch) > max_epoch:
                max_epoch = int(epoch)

        if max_epoch >= 0:
            args.resume = os.path.join(args.output_dir, f"checkpoint-{max_epoch}.pth")
            print(f"Resuming from {args.resume}")    

    torch.distributed.barrier()
    main(args)