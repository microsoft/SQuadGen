from torch.utils import data
import copy
import torch
import numpy as np
import os
import json
from collections.abc import Sequence, Mapping

from .utils import *
from squadgen.util.util import load_data

class ToTensor(object):
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            # note that str is also a kind of sequence, judgement should before sequence
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        elif isinstance(data, Mapping):
            result = {sub_key: self(item) for sub_key, item in data.items()}
            return result
        elif isinstance(data, Sequence):
            result = [self(item) for item in data]
            return result
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")

def load_filelist(folder, data_filter_name="", dataset_type="train"):

    if data_filter_name == "":
        p = os.path.join(folder, f"_{dataset_type}_filelist.txt")
    else:
        p = os.path.join(folder, data_filter_name, f"_{dataset_type}_filelist.txt")

    with open(p, "r") as f:
        lines = f.read().splitlines()
        return lines

def load_one_filelist(folder, data_filter_name, dataset_type, start_file, end_file, repeat_times, is_shuffle):
    filelist = load_filelist(folder, data_filter_name, dataset_type)

    print(f"{folder}, original data size:", len(filelist))

    if is_shuffle:
        generator = torch.Generator().manual_seed(0)
        filelist_idx = torch.randperm(len(filelist), generator=generator).tolist()
        filelist = [filelist[i] for i in filelist_idx]

    if end_file != -1:
        end_file = min(end_file, len(filelist))
    else:
        end_file = len(filelist)

    filelist = filelist[start_file:end_file]
    print(f"dataset_type: {dataset_type}, filelist data selection interval:", start_file, end_file, "repeat times:", repeat_times, "len(filelist):", len(filelist))

    return filelist

class QuadDataset(data.Dataset):
    def __init__(self, folder: str,
                 n_sample_surface=1024, n_sample_surface_query=1024,
                 n_sample_near=1024, n_sample_global=1024,
                 deterministic=False,
                 dataset_type="train", repeat_times=1,
                 start_file=0, end_file=-1,
                 data_filter_name="",
                 is_shuffle=False,
                 num_latents_list=[],
                 labeling_fn="",
                 epoch_len=-1,
                 ):
        super().__init__()

        if not isinstance(dataset_type, list):
            dataset_type = [dataset_type]

        if labeling_fn != "":
            assert len(dataset_type) == 1

        print("dataset_type list:", dataset_type)

        filelist_label = []
        filelist = []
        for dt in dataset_type:
            tmp = load_one_filelist(folder, data_filter_name, dt, start_file, end_file, 1, is_shuffle)
            filelist_label.extend([dt] * len(tmp))
            filelist.extend(tmp)
            print(f"filelist {dt} size:", len(tmp))

        if epoch_len <= 0:
            epoch_len = None

        self.filelist = filelist
        self.filelist_label = filelist_label
        self.repeat_times = repeat_times
        self.folder = folder
        self.tudf_threshold = 0.08
        self.n_sample_surface = n_sample_surface
        self.n_sample_surface_query = n_sample_surface_query
        self.n_sample_near = n_sample_near
        self.n_sample_global = n_sample_global
        self.deterministic = deterministic
        self.num_latents_list = list(set([x for x in num_latents_list if x > 0]))
        self.labeling_fn = labeling_fn
        self.epoch = 0
        self.epoch_len = epoch_len

        print("filelist size:", len(self.filelist), "repeat_times:", self.repeat_times)

        self.generator_list = [torch.Generator(device="cpu").manual_seed(i) for i in range(1000)]
        self.generator_np_list = [np.random.default_rng(i) for i in range(1000)]

        if self.labeling_fn != "":
            assert self.filelist[0].startswith("/")

            self.generator_select = torch.Generator(device="cpu").manual_seed(12345)

            with open(os.path.join(folder, data_filter_name, labeling_fn), "r") as f:
                clusters = json.load(f)

            config = {}
            dataset_dict = {}

            for idx, cluster in enumerate(clusters):
                config[idx] = {"ratio": cluster[0]}
                dataset_dict[idx] = cluster[1]

            self.dataset_dict = dataset_dict
            self.balance_config = config

            sum_rate = sum(config[x]["ratio"] for x in config)
            assert abs(sum_rate - 1) < 1e-6, f"sum of ratio should be 1, but {sum_rate}"

            print(f"dataset number (sorted): {sorted([len(self.dataset_dict[k]) for k in self.dataset_dict])}")
            print(f"dataset number: {[len(self.dataset_dict[k]) for k in self.dataset_dict]}")

        if dataset_type == ["train"]:
            self.show_data_format()

        print("Dataset length:", len(self))

    def show_data_format(self):
        fn = self.get_filename(0, "npz")
        data = load_data(fn)
        print("Raw data npz format:", data.keys())
        print(self.filelist[0])
        for key in data.keys():
            print(key, data[key].shape, data[key].dtype)

        print("Preprocess data npz format:", data.keys())
        data = self.__getitem__(0)
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                print(key, "torch.Tensor", data[key].shape, data[key].dtype)
            elif isinstance(data[key], list):
                print(key, "list", len(data[key]), data[key])
            else:
                print(key, type(data[key]))

    def get_filename(self, idx, filetype):
        try:
            p = self.filelist[idx]
        except:
            print("idx", idx, len(self.filelist), len(self), self.epoch_len)
            raise
        return p

    def __len__(self):
        if self.epoch_len is not None:
            return self.epoch_len * self.repeat_times
        return len(self.filelist) * self.repeat_times

    def transform_data(self, data_dict):
        # data_dict:
        # points: xyz and color, C=4, xyz in [-1, 1], center is (0, 0, 0), color in {-1, 0, 1}
        # normal: shape of normal is [N, 3], float32
        # sizing: shape of sizing is [N, 1], float32
        # offset: shape of offset is [N, 3], float32

        ans = ToTensor()(data_dict)

        ans_ret = {}
        for k in ans.keys():
            if k in ["on_surface"]:
                ans_ret[k] = ans[k].bool()
            elif k.startswith("fps_idx"):
                ans_ret[k] = ans[k].to(torch.long)
            elif k == "batch_id":
                ans_ret[k] = ans[k].to(torch.int32)
            elif not isinstance(ans[k], torch.Tensor):
                ans_ret[k] = ans[k]
            else:
                ans_ret[k] = ans[k].to(torch.float32)
        return ans_ret

    def select_class(self, ratio_target, balance_config, dataset_dict):
        for k in balance_config:
            if ratio_target < balance_config[k]["ratio"]:
                break
            ratio_target -= balance_config[k]["ratio"]
        generator = self.generator_select if self.deterministic else None
        idx = torch.randint(0, len(dataset_dict[k]), (1, ), generator=generator).item()
        return dataset_dict[k][idx], k

    def __getitem__(self, index_raw):

        class_id = -1
        if self.labeling_fn == "":
            index = index_raw // self.repeat_times
        else:
            index, class_id = self.select_class(index_raw / len(self), self.balance_config, self.dataset_dict)
            if isinstance(index, list):
                index = index[0]

        while True:
            try:
                input_fn = self.get_filename(index, "npz")
                sample = load_data(input_fn)

                ans = {}
                from data_tools.test_load_new_format import load_patch_data_reformat

                load_func = load_patch_data_reformat

                num_latents_list = copy.deepcopy(self.num_latents_list)
                num_latents_list += [4*x for x in self.num_latents_list]

                np.seterr(all='raise')
                ans = load_func(
                    npzfile=input_fn,
                    num_surface=self.n_sample_surface,
                    num_surface_query=self.n_sample_surface_query,
                    file=sample,
                    fps_num_list=num_latents_list,
                    generator=self.generator_np_list[100] if not self.deterministic else np.random.default_rng(0),
                    fps_return_type="random" if not self.deterministic else "first",
                    is_add_noise=not self.deterministic,
                )
                flag = 0
                for k in ans:
                    # if inf, -inf, nan
                    if np.isnan(ans[k]).any() or np.isinf(ans[k]).any():
                        flag = 1
                        break
                if flag:
                    print(f"Error: {k} has nan or inf")
                    index = (index + 1) % len(self.filelist)
                    # raise ValueError(f"{k} has nan or inf")
                    continue

                udf_query = torch.zeros((self.n_sample_surface_query, 1), dtype=torch.float32)
                ans["udf_query"] = udf_query

                ans = ToTensor()(ans)

                ans = self.transform_data(ans)

                ans["checker_sizing"] = ans["checker_sizing"].numpy().tolist()
                if "invT" in ans:
                    ans["invT"] = ans["invT"].cpu().numpy()


                ans["index"] = index
                ans["data_type"] = self.filelist_label[index]
                ans["class_id"] = class_id
                ans["epoch"] = self.epoch

                return ans
                break
            except Exception as e:
                # print(f"Error at file {self.get_filename(index, 'npz')} with error: {e}")
                import traceback
                tb = traceback.extract_tb(e.__traceback__)[-1]
                print(f"Error at file {self.get_filename(index, 'npz')} with error: {e}, line: {tb.lineno}, code: {tb.line}")
                index = (index + 1) % len(self.filelist)
                # raise e

class MergePointClouds:
    def __call__(self, data_list: list):
        # (N, C+1), C is the feature dimension, the last column is the batch index
        keys_list = [
            "xyz", "color",
            "xyz_near", "sdf_near", "udf_near",
            "xyz_global", "sdf_global", "udf_global",
            "normal", "sizing",
            "offset", "offset1", "offset2",
            "offset_div_sizing", "offset1_div_sizing", "offset2_div_sizing",
            "on_surface",
        ]
        ans_dict = {}
        batch_id_all = []
        index_list = []
        data_type_list = []
        for i, data in enumerate(data_list):
            index = data["index"]
            batch_index = torch.full((data["xyz"].shape[0], ), i, dtype=torch.int32)
            batch_id_all.append(batch_index)
            for key in data:
                if key not in ans_dict:
                    ans_dict[key] = []
                ans_dict[key].append(data[key])
            index_list.append(index)
            data_type_list.append(data["data_type"])

        batch_id_all = torch.cat(batch_id_all, dim=0)
        for key in ans_dict.keys():
            if len(ans_dict[key]) > 0 and isinstance(ans_dict[key][0], torch.Tensor):
                ans_dict[key] = torch.stack(ans_dict[key], dim=0)
        # data: (N, C=3+1+1), C: xyz, color, batch_idx

        ans = {"batch_size": len(data_list), "index_list": index_list, "data_type_list": data_type_list,
               **ans_dict}
        return ans

def get_shapenet_sparsity_dataset(folder,
                                  dataset_type="train",
                                  repeat_times=1,
                                  start_file=0,
                                  end_file=-1,
                                  data_filter_name="",
                                  is_shuffle=False,
                                  n_sample_surface=1024, n_sample_surface_query=1024,
                                  n_sample_near=1024, n_sample_global=1024,
                                  deterministic=False,
                                  num_latents_list=[],
                                  labeling_fn="",
                                  epoch_len=-1,
                                  ):
    collate_batch = MergePointClouds()

    dataset = QuadDataset(
        folder=folder,
        n_sample_surface=n_sample_surface, n_sample_surface_query=n_sample_surface_query,
        n_sample_near=n_sample_near, n_sample_global=n_sample_global,
        deterministic=deterministic,
        dataset_type=dataset_type, repeat_times=repeat_times,
        start_file=start_file, end_file=end_file,
        data_filter_name=data_filter_name,
        is_shuffle=is_shuffle,
        num_latents_list=num_latents_list,
        labeling_fn=labeling_fn,
        epoch_len=epoch_len,
        )

    return dataset, collate_batch

