from functools import wraps

import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from timm.models.vision_transformer import Attention

from timm.models.layers import DropPath
from .utils import KEY_DIM_DICT
from .utils import *

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None, is_context_norm = 1, is_x_norm = 1):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim) if is_x_norm else nn.Identity()
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) and is_context_norm else None
        self.is_context_norm = is_context_norm
        self.is_x_norm = is_x_norm

    def forward_part(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context) and self.is_context_norm:
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return x, kwargs, self.fn

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context) and self.is_context_norm:
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, drop_path_rate = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, **kwargs):
        return self.drop_path(self.net(x))

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, drop_path_rate = 0.0, qknorm=0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.context_dim = context_dim
        self.qknorm = qknorm

        self.q_norm = nn.LayerNorm(dim_head) if qknorm else nn.Identity()
        self.k_norm = nn.LayerNorm(dim_head) if qknorm else nn.Identity()

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def get_attn_map(self, x, context = None, mask = None, **kwargs):
        h = self.heads
        assert h == 1

        x = x.detach()
        context = default(context, x).detach()

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q = self.q_norm(q) # [B*h, N_q, D]
        k = self.k_norm(k) # [B*h, N_kv, D]

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale # [B*h, N_q, N_kv]
        
        if exists(mask):
            # mask: [b, n_q, n_kv], bool
            mask = rearrange(mask, 'b n m -> b 1 n m')
            mask = repeat(mask, 'b 1 n m -> (b h) n m', h=h) # [B*h, N_q, N_kv]
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)
        
        attn = sim.softmax(dim = -1)
        # [B*h, N_q, N_kv] -> [B, h, N_q, N_kv]
        attn = rearrange(attn, '(b h) n m -> b h n m', h=h)
        attn = attn[:, 0, ...]
        return attn

    def forward(self, x, context = None, mask = None, **kwargs):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q = self.q_norm(q)
        k = self.k_norm(k)

        if mask is not None:
            assert mask.shape[1] == q.shape[1] and mask.shape[2] == k.shape[1], f"{mask.shape} {q.shape} {k.shape}"


        if torch.__version__ < "2.1":
            out = torch.nn.functional.scaled_dot_product_attention(
                q*self.scale*(q.size(-1)**0.5), k, v, 
                dropout_p=0.0,
                attn_mask=mask,
            )
        else:
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                dropout_p=0.0, 
                scale=self.scale,
                attn_mask=mask,
            )                

        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.drop_path(self.to_out(out))


class PointEmbed(nn.Module):
    def __init__(self, dim=128, in_keys_list=[]):
        super().__init__()

        self.dim = dim
        self.mlp_pos = nn.Linear(3, dim//2, bias=False)
        self.mlp_pos_post = nn.Linear(dim//2*2+3, dim)

        num_concat = dim
        for k in in_keys_list:
            in_channels = KEY_DIM_DICT[k]
            out_channels = dim
            num_concat += out_channels

            mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
            )
            setattr(self, f"mlp_{k}", mlp)

        self.mlp = nn.Linear(num_concat, dim)

    def get_xyz_pos_embed(self, xyz):
        freqs = self.mlp_pos(xyz) * 2 * np.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat([xyz, fouriered], dim=-1)
        fouriered = self.mlp_pos_post(fouriered)
        return fouriered

    def forward(self, input, context_dict):
        # input: B x N x 3
        embed = self.get_xyz_pos_embed(input) # B x N x C
        for k, v in context_dict.items():
            mlp = getattr(self, f"mlp_{k}")
            x = mlp(v)
            embed = torch.cat([embed, x], dim=2)
        embed = self.mlp(embed)
        return embed

    def query(self, input):
        return self.get_xyz_pos_embed(input)

def check_for_nans(tensor):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        tensor = torch.where(torch.isnan(tensor) | torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor

class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -10.0, 10.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                ans = 0.5 * torch.mean(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
                return check_for_nans(ans)
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

class KLAutoEncoder(nn.Module):
    def __init__(
        self,
        in_keys_list,
        out_keys_list,
        *,
        out_weights_list=None,
        depth=24,
        dim=512,
        queries_dim=512,
        num_inputs = 2048,
        num_latents = 512,
        latent_dim = 64,
        heads = 8,
        is_learnable_latents = 0,
        dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False,
        is_context_norm = 1,
        is_dec_cross_norm=1,
        is_enc_cross_norm=0,
        drop_path_rate=0.1,
        attn_depth_enc=0,
        qknorm=0,
        use_geom_code_as_fps=0,
        use_geom_code_and_pe_as_fps=0,
        gcolor_grad_norm_in=0,
    ):
        super().__init__()

        self.depth = depth

        self.num_latents = num_latents
        self.in_keys_list = in_keys_list
        self.out_keys_list = out_keys_list
        self.out_weights_list = out_weights_list
        self.in_keys_list_mlp = list(set([x.replace("_edge", "") for x in in_keys_list]))
        self.out_keys_list_mlp = list(set([x.replace("_near", "").replace("_global", "").replace("_edge", "") for x in out_keys_list]))
        self.in_keys_list_mlp = sorted(self.in_keys_list_mlp)
        self.out_keys_list_mlp = sorted(self.out_keys_list_mlp)
        self.is_learnable_latents = is_learnable_latents
        self.is_context_norm = is_context_norm
        self.attn_depth_enc = attn_depth_enc
        self.qknorm = qknorm
        self.use_geom_code_as_fps = use_geom_code_as_fps
        self.use_geom_code_and_pe_as_fps = use_geom_code_and_pe_as_fps
        self.gcolor_grad_norm_in = gcolor_grad_norm_in

        dim_in = dim

        if is_context_norm:
            enc_norm_func = lambda *args, **kwargs: PreNorm(*args, **kwargs, is_context_norm=is_enc_cross_norm)
            dec_norm_func = PreNorm
            dec_norm_func_last = lambda *args, **kwargs: PreNorm(*args, **kwargs, is_context_norm=1, is_x_norm=is_dec_cross_norm)
        else:
            enc_norm_func = lambda *args, **kwargs: PreNorm(*args, **kwargs, is_context_norm=is_enc_cross_norm)
            dec_norm_func = lambda *args, **kwargs: PreNorm(*args, **kwargs, is_context_norm=0)
            dec_norm_func_last = lambda *args, **kwargs: PreNorm(*args, **kwargs, is_context_norm=0, is_x_norm=is_dec_cross_norm)

        self.cross_attend_blocks = nn.ModuleList([
            enc_norm_func(dim_in, Attention(dim_in, dim_in, heads = 1, dim_head = dim_in, qknorm=self.qknorm), context_dim = dim_in),
            enc_norm_func(dim_in, FeedForward(dim_in)),
        ])

        if self.attn_depth_enc > 0:
            get_latent_attn = lambda: enc_norm_func(dim, Attention(dim, heads = heads, dim_head = dim_head, drop_path_rate=drop_path_rate, qknorm=self.qknorm))
            get_latent_ff = lambda: enc_norm_func(dim, FeedForward(dim, drop_path_rate=drop_path_rate))
            get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

            self.layers_enc = nn.ModuleList([])
            cache_args = {'_cache': weight_tie_layers}
            
            for i in range(self.attn_depth_enc):
                self.layers_enc.append(nn.ModuleList([
                    get_latent_attn(**cache_args),
                    get_latent_ff(**cache_args)
                ]))

        self.point_embed = PointEmbed(dim=dim, in_keys_list=self.in_keys_list_mlp)

        get_latent_attn = lambda: dec_norm_func(dim, Attention(dim, heads = heads, dim_head = dim_head, drop_path_rate=drop_path_rate, qknorm=self.qknorm))
        get_latent_ff = lambda: dec_norm_func(dim, FeedForward(dim, drop_path_rate=drop_path_rate))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            x = [get_latent_attn(**cache_args)]
            x.append(get_latent_ff(**cache_args))
            self.layers.append(nn.ModuleList(x))

        if self.is_learnable_latents:
            # claim n_latents learnable latents
            self.latents_input = nn.Parameter(torch.randn(num_latents, dim))

        self.decoder_cross_attn = dec_norm_func_last(queries_dim, Attention(queries_dim, dim, heads = 1, dim_head = dim, qknorm=self.qknorm), context_dim = dim)
        self.decoder_ff = dec_norm_func_last(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.to_outputs = nn.ModuleDict()
        for k in self.out_keys_list_mlp:
            net = nn.Sequential(
                nn.Linear(queries_dim, queries_dim),
                nn.ReLU(),
                nn.Linear(queries_dim, queries_dim),
                nn.ReLU(),
                nn.Linear(queries_dim, KEY_DIM_DICT[k]),
            )
            self.to_outputs[k] = net

        self.proj = nn.Linear(latent_dim, dim)

        self.mean_fc = nn.Linear(dim, latent_dim)
        self.logvar_fc = nn.Linear(dim, latent_dim)

        if self.use_geom_code_and_pe_as_fps:
            self.proj_geom_code_fps = nn.Linear(dim, dim)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # pos embedding weight
        nn.init.normal_(self.point_embed.mlp_pos.weight, std=1)
        for k in self.in_keys_list_mlp:
            nn.init.normal_(getattr(self.point_embed, f"mlp_{k}")[0].weight, std=0.02)

        # Zero-out output layers:
        for k in self.out_keys_list_mlp:
            nn.init.constant_(self.to_outputs[k][-1].weight, 0)
            nn.init.constant_(self.to_outputs[k][-1].bias, 0)

    def get_input(self, batch):
        n = self.num_latents
        k_suff = f"_fps_{4*n}"
        pc_input = batch[f"xyz{k_suff}"]
        pc_context_dict = {}
        for k in self.in_keys_list_mlp:
            x = batch[f"{k}{k_suff}"]
            if k in ["gcdf", "gdcdf"] and self.gcolor_grad_norm_in:
                x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
            pc_context_dict[k] = x
        if len(pc_input.shape) == 3:
            # convert to [B, N, m, C]
            pc_input = pc_input[:, :, None, ...]
            pc_context_dict = {k: v[:, :, None, ...] for k, v in pc_context_dict.items()}
        ans = {
            "pc": {"input": pc_input, "context_dict": pc_context_dict},
        }
        if self.is_learnable_latents == 0:
            fps_input = batch[f"xyz_fps_{n}"]
            fps_context_dict = {}
            for k in self.in_keys_list_mlp:
                x = batch[f"{k}_fps_{n}"]
                if k in ["gcdf", "gdcdf"] and self.gcolor_grad_norm_in:
                    x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
                fps_context_dict[k] = x
            ans.update({
                "fps": {"input": fps_input, "context_dict": fps_context_dict},
            })
        return ans

    def get_enc_attn_map(self, batch, geom_code_fps=None):
        data = self.get_input(batch)

        if self.is_learnable_latents:
            sampled_pc_embeddings = self.latents_input[None, ...].repeat(batch["xyz"].shape[0], 1, 1)
        elif self.use_geom_code_as_fps:
            assert geom_code_fps is not None
            if self.use_geom_code_and_pe_as_fps:
                x_pe = self.proj_geom_code_fps(geom_code_fps)
                sampled_pc_embeddings = x_pe + self.point_embed(**data["fps"])
            else:
                x_pe = geom_code_fps
                sampled_pc_embeddings = geom_code_fps
        else:      
            sampled_pc_embeddings = self.point_embed(**data["fps"]) # [B, N, D]
        
        idx_group = 0
        pc_current = {
            "input": data["pc"]["input"][:, :, idx_group, :],
            "context_dict": {k: v[:, :, idx_group, :] for k, v in data["pc"]["context_dict"].items()}
        }
        pc_embeddings = self.point_embed(**pc_current)
        x = sampled_pc_embeddings

        print(f"sampled_pc_embeddings.shape={sampled_pc_embeddings.shape}, pc_embeddings.shape={pc_embeddings.shape}")

        cross_attn, cross_ff = self.cross_attend_blocks

        x, kwargs, fn = cross_attn.forward_part(sampled_pc_embeddings, context = pc_embeddings)
        return fn.get_attn_map(x, **kwargs)

    def encode(self, batch, is_mode=False, geom_code_fps=None):
        data = self.get_input(batch)

        x_pe = None
        if self.is_learnable_latents:
            sampled_pc_embeddings = self.latents_input[None, ...].repeat(batch["xyz"].shape[0], 1, 1)
        elif self.use_geom_code_as_fps:
            assert geom_code_fps is not None
            if self.use_geom_code_and_pe_as_fps:
                x_pe = self.proj_geom_code_fps(geom_code_fps)
                sampled_pc_embeddings = x_pe + self.point_embed(**data["fps"])
            else:
                x_pe = geom_code_fps
                sampled_pc_embeddings = geom_code_fps
        else:      
            sampled_pc_embeddings = self.point_embed(**data["fps"]) # [B, N, D]
        
        results_list = []
        for idx_group in range(data["pc"]["input"].shape[2]):
            pc_current = {
                "input": data["pc"]["input"][:, :, idx_group, :],
                "context_dict": {k: v[:, :, idx_group, :] for k, v in data["pc"]["context_dict"].items()}
            }
            pc_embeddings = self.point_embed(**pc_current)

            cross_attn, cross_ff = self.cross_attend_blocks

            x = sampled_pc_embeddings

            x = cross_attn(x, context = pc_embeddings) + x
            x = cross_ff(x) + x

            if self.attn_depth_enc > 0:
                for idx, block in enumerate(self.layers_enc):
                    self_attn, self_ff = block
                    x = self_attn(x) + x
                    x = self_ff(x) + x
            
            # x.shape == [B, N, D]

            results_list.append(x)
        
        x = torch.stack(results_list, dim=2) # [B, N, m, D]
        # max pooling along dim=2
        x = x.max(dim=2)[0] # [B, N, D]

        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mean, logvar)
        if is_mode:
            x = posterior.mode()
        else:
            x = posterior.sample()
        kl = posterior.kl()

        return {
            "input": data,
            "kl": kl,
            "latent": x,
            "posterior": posterior,
        }
    
    def get_pn(self, name, n=1):
        func = lambda x, **kwargs: x
        if n == 1:
            return func
        return [func for _ in range(n)]

    def return_last_features(self, x, context=None):
        x = self.proj(x)
        for idx, block in enumerate(self.layers):
            self_attn, self_ff = block
            pn0, pn1 = self.get_pn(f"dec_{idx}", 2)
            x = pn0(self_attn(x)) + x
            x = pn1(self_ff(x)) + x
        return x

    def return_last_features_and_norm(self, x, context=None):
        x = self.return_last_features(x, context=context)

        if exists(self.decoder_cross_attn.norm_context) and self.decoder_cross_attn.is_context_norm:
            x = self.decoder_cross_attn.norm_context(x)
        return x

    def get_dec_attn_map(self, x, batch, context=None):

        mark = {
            "surface": 0,
            "near": 0,
            "global": 0,
            "edge": 0,
        }
        for k in self.out_keys_list:
            if k.endswith("_near"):
                mark["near"] = 1
            elif k.endswith("_global"):
                mark["global"] = 1                
            elif k.endswith("_edge"):
                mark["edge"] = 1
            else:
                mark["surface"] = 1

        interval = {}
        cur = 0
        queries = []
        for k in mark:
            if mark[k]:
                if k == "surface":
                    k_ = "xyz_query"
                elif k == "near":
                    k_ = "xyz_near"
                elif k == "global":
                    k_ = "xyz_global"
                elif k == "edge":
                    k_ = "xyz_queryedge"
                queries.append(batch[k_])
                interval[k] = [cur, cur+batch[k_].shape[1]]
                cur += batch[k_].shape[1] 
        queries = torch.cat(queries, dim=1)

        x = self.return_last_features(x, context=context)

        queries_embeddings = self.point_embed.query(queries)
        print(f"get_dec_attn_map: x.shape={x.shape}, queries_embeddings.shape={queries_embeddings.shape}")
        x, kwargs, fn = self.decoder_cross_attn.forward_part(queries_embeddings, context=x)
        return fn.get_attn_map(x, **kwargs)

    def query_last_features(self, x, queries):
        # cross attend from decoder queries to latents
        queries_embeddings = self.point_embed.query(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context=x)
     
        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)
        
        output = {}
        for k in self.out_keys_list_mlp:
            x = self.to_outputs[k](latents)
            if k == "normal":
                x = F.normalize(x, dim=-1) # normalize normal
            elif k in ["offset"]:
                x = x / 7
            elif k in ["offset1", "offset2"]:
                x = x / 3.5
            output[k] = x
        return output

    def decode(self, x, queries, context=None):
        x = self.return_last_features(x, context=context)
        return self.query_last_features(x, queries=queries)

    def decode_batch(self, x, batch, context=None):
        
        mark = {
            "surface": 0,
            "near": 0,
            "global": 0,
            "edge": 0,
        }
        for k in self.out_keys_list:
            if k.endswith("_near"):
                mark["near"] = 1
            elif k.endswith("_global"):
                mark["global"] = 1                
            elif k.endswith("_edge"):
                mark["edge"] = 1
            else:
                mark["surface"] = 1

        interval = {}
        cur = 0
        queries = []
        for k in mark:
            if mark[k]:
                if k == "surface":
                    k_ = "xyz_query"
                elif k == "near":
                    k_ = "xyz_near"
                elif k == "global":
                    k_ = "xyz_global"
                elif k == "edge":
                    k_ = "xyz_queryedge"
                queries.append(batch[k_])
                interval[k] = [cur, cur+batch[k_].shape[1]]
                cur += batch[k_].shape[1] 
        queries = torch.cat(queries, dim=1)

        out = self.decode(x, queries, context=context)

        output = {}
        for k in self.out_keys_list:
            k_ = k.replace("_near", "").replace("_global", "").replace("_edge", "")
            if k.endswith("_near"):
                s = "near"
            elif k.endswith("_global"):
                s = "global"                
            elif k.endswith("_edge"):
                s = "edge"
            else:
                s = "surface"
            output[k] = out[k_][:, interval[s][0]:interval[s][1], ...]
        
        return output

def create_vae_from_config(config):
    print(config)
    out_weights_list = list(config.out_weights_list) if hasattr(config, "out_weights_list") else [1 for _ in config.out_keys_list]
    if sum(out_weights_list) == 0: # this model is not used
        return None

    if hasattr(config, "is_context_norm"):
        is_context_norm = config.is_context_norm
    else:
        is_context_norm = 1

    if is_context_norm:
        is_dec_cross_norm = 1
        is_enc_cross_norm = 1
    else:
        is_dec_cross_norm = config.is_dec_cross_norm if hasattr(config, "is_dec_cross_norm") else 1
        is_enc_cross_norm = config.is_enc_cross_norm if hasattr(config, "is_enc_cross_norm") else 0

    drop_path_rate = config.drop_path_rate if hasattr(config, "drop_path_rate") else 0.1

    attn_depth_enc = config.attn_depth_enc if hasattr(config, "attn_depth_enc") else 0

    qknorm = config.qknorm if hasattr(config, "qknorm") else 0

    use_geom_code_as_fps = 0 if not hasattr(config, "use_geom_code_as_fps") else config.use_geom_code_as_fps
    use_geom_code_and_pe_as_fps = 0 if not hasattr(config, "use_geom_code_and_pe_as_fps") else config.use_geom_code_and_pe_as_fps
    if use_geom_code_and_pe_as_fps: assert use_geom_code_as_fps

    gcolor_grad_norm_in = 0 if not hasattr(config, "gcolor_grad_norm_in") else config.gcolor_grad_norm_in
    model = KLAutoEncoder(
        in_keys_list=config.in_keys_list,
        out_keys_list=config.out_keys_list,
        out_weights_list=out_weights_list,
        depth=config.attn_depth,
        dim=config.dim,
        queries_dim=config.dim,
        num_latents = config.num_latents,
        latent_dim = config.latent_dim,
        heads = 8,
        dim_head = 64,
        is_learnable_latents=config.is_learnable_latents if hasattr(config, "is_learnable_latents") else 0,
        is_context_norm=is_context_norm,
        is_dec_cross_norm=is_dec_cross_norm,
        is_enc_cross_norm=is_enc_cross_norm,
        drop_path_rate=drop_path_rate,
        attn_depth_enc=attn_depth_enc,
        qknorm=qknorm,
        use_geom_code_as_fps=use_geom_code_as_fps,
        use_geom_code_and_pe_as_fps=use_geom_code_and_pe_as_fps,
        gcolor_grad_norm_in=gcolor_grad_norm_in,
    )
    return model
