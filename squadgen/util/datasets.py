from torch.utils.data import DataLoader, DistributedSampler

import squadgen.util.misc as misc
from squadgen.network.data_loader import get_shapenet_sparsity_dataset

def get_dataloader(args, num_latents_list):

    def get_arg(args, name, default):
        return getattr(args, name, default)

    def get_train(args, num_latents_list):
        dataset, collate_fn = get_shapenet_sparsity_dataset(args.dataset_folder,
                                    n_sample_surface=get_arg(args, "n_sample_surface", 1024),
                                                            n_sample_surface_query=get_arg(args, "n_sample_surface_query", 1024),
                                    n_sample_near=get_arg(args, "n_sample_near", 0),
                                                            n_sample_global=get_arg(args, "n_sample_global", 0),
                                                            deterministic=args.training_determistic,
                                                            data_filter_name=args.data_filter_name,
                                                            repeat_times=args.training_data_repeat_times,
                                                            num_latents_list=num_latents_list,
                                                            labeling_fn=args.labeling_fn if hasattr(args, "labeling_fn") else "",
                                                            epoch_len=args.training_epoch_len if hasattr(args, "training_epoch_len") else 450000,
                                                            )
        return dataset, collate_fn

    dataset_train, collate_fn_train = get_train(args, num_latents_list)

    def get_val(args, num_latents_list):
        if hasattr(args, "val_dataset_type_list"):
            dataset_type=args.val_dataset_type_list.split(",")
        else:
            dataset_type=["test", "train"]
        max_num=args.val_num
        repeat_times = 1
        is_shuffle=True
        start_file = 0

        print(f"val data, dataset_type: {dataset_type}, max_num: {max_num}, start_file: {start_file}")

        dataset, collate_fn = get_shapenet_sparsity_dataset(args.dataset_folder,
                                                            n_sample_surface=get_arg(args, "n_sample_surface_val", 1024),
                                                            n_sample_surface_query=get_arg(args, "n_sample_surface_query_val", 1024),
                                                            n_sample_near=get_arg(args, "n_sample_near_val", 0),
                                                            n_sample_global=get_arg(args, "n_sample_global_val", 0),
                                                            deterministic=True,
                                                            dataset_type=dataset_type,
                                                            start_file=start_file,
                                                            end_file=start_file+max_num,
                                                            data_filter_name=args.data_filter_name,
                                                            is_shuffle=is_shuffle,
                                                            repeat_times=repeat_times,
                                                            num_latents_list=num_latents_list,
                                                            )
        return dataset, collate_fn

    dataset_val, collate_fn_val = get_val(args, num_latents_list)


    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
    if len(dataset_val) % num_tasks != 0:
        print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                'This will slightly alter validation results as extra duplicate entries are added to achieve '
                'equal num of samples per-process.')
    sampler_val = DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)  # shuffle=True to reduce monitor bias

    print("Sampler_val = %s" % str(sampler_val))

    data_loader_train = DataLoader(
        dataset_train, 
        sampler=sampler_train,
        collate_fn=collate_fn_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    val_batch_size = min(args.batch_size, args.val_num // num_tasks)

    data_loader_val = DataLoader(
        dataset_val, 
        sampler=sampler_val,
        collate_fn=collate_fn_val,
        batch_size=val_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    print("dataset_val.filelist_label:", dataset_val.filelist_label)
    for idx_batch, batch in enumerate(data_loader_val):
        data_type_list = batch["data_type_list"]
        index_list = batch["index_list"]
        print(f"batch_{idx_batch}:", data_type_list, index_list)

    for batch in data_loader_val:
        data_type_list = batch["data_type_list"]
        for data_type in data_type_list:
            assert data_type == data_type_list[0]

    return {
        "train_loader": data_loader_train,
        "val_loader": data_loader_val,
        "train": dataset_train,
        "val": dataset_val,
    }
