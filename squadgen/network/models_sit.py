# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Mlp, use_fused_attn
from torch.jit import Final

from typing import Type

from .utils import *

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=10000):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size, self.max_period)
        t_emb = self.mlp(t_freq)
        return t_emb

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # [3, B, num_heads, N, head_dim]
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PointEmbed(nn.Module):
    def __init__(self, dim=128, in_keys_list=()):
        super().__init__()

        self.dim = dim
        self.mlp_pos = nn.Linear(3, dim//2, bias=False)
        self.mlp_pos_post = nn.Linear(dim//2*2+3, dim)

        num_concat = dim
        for k in in_keys_list:
            mlp = nn.Sequential(
                nn.Linear(KEY_DIM_DICT[k], dim),
            )
            setattr(self, f"mlp_{k}", mlp)
            num_concat += dim

        self.mlp = nn.Linear(num_concat, dim)

    def get_xyz_pos_embed(self, xyz):
        freqs = self.mlp_pos(xyz) * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat([xyz, fouriered], dim=-1)
        fouriered = self.mlp_pos_post(fouriered)
        return fouriered

    def forward(self, input, context_dict):
        # input: B x N x 3
        embed = self.get_xyz_pos_embed(input) # B x N x C
        for k, v in context_dict.items():
            mlp = getattr(self, f"mlp_{k}")
            embed = torch.cat([embed, mlp(v)], dim=2)
        embed = self.mlp(embed)
        return embed


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        n_latents=512,
        mlp_ratio=4.0,
        learn_sigma=False,
        in_channels = 32,
        depth = 24,
        n_heads = 8,
        d_head = 64,
        condition_global_keys_list = [],
        in_channels_cond = 64,
        is_geom_last_feature= 0,
        is_fps_cond = 0,
        qk_norm = 0,
        is_pe_latent = 0,
        is_pecode_latent = 0,
        is_pecode_latent_pe = 0,
        target_mu = None,
        target_std = None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.in_channels_cond = in_channels_cond
        self.hidden_size = n_heads * d_head
        self.channels = in_channels
        self.condition_global_keys_list = condition_global_keys_list
        self.n_latents = n_latents
        self.is_geom_last_feature = is_geom_last_feature
        self.is_fps_cond = is_fps_cond
        self.is_pe_latent = is_pe_latent
        self.is_pecode_latent = is_pecode_latent
        self.is_pecode_latent_pe = is_pecode_latent_pe
        self.target_mu = target_mu
        self.target_std = target_std

        hidden_size = self.hidden_size
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        for key in condition_global_keys_list:
            setattr(self, f"proj_{key}", TimestepEmbedder(hidden_size, max_period=32))

        self.input_proj = nn.Linear(self.in_channels, hidden_size, bias=True)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, n_heads, mlp_ratio=mlp_ratio, qk_norm=qk_norm, norm_layer=nn.RMSNorm) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels)

        if self.is_pe_latent:
            self.pe_embed_list = nn.ModuleList([
                PointEmbed(dim=hidden_size, in_keys_list=["normal"]) for _ in range(depth)
            ])
        elif self.is_pecode_latent:
            self.pe_embed_list = nn.ModuleList([
                nn.Linear(in_channels_cond, hidden_size, bias=False) for _ in range(depth)
            ])
            if self.is_pecode_latent_pe:
                self.pe_embed_list2 = nn.ModuleList([
                    PointEmbed(dim=hidden_size, in_keys_list=["normal"]) for _ in range(depth)
                ])  

        self.initialize_weights()

    def get_target_stats(self, device, dtype=torch.float32):
        assert isinstance(self.target_mu, list) and len(self.target_mu) == self.channels
        assert isinstance(self.target_std, list) and len(self.target_std) == self.channels
        target_mu = torch.tensor(self.target_mu, device=device, dtype=dtype)
        target_std = torch.tensor(self.target_std, device=device, dtype=dtype)
        return target_mu, target_std

    def normalize_latents(self, latents):
        target_mu, target_std = self.get_target_stats(latents.device, latents.dtype)
        return (latents - target_mu[None, None, :]) / target_std[None, None, :]

    def denormalize_latents(self, latents):
        target_mu, target_std = self.get_target_stats(latents.device, latents.dtype)
        return latents * target_std[None, None, :] + target_mu[None, None, :]

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize label embedding table:
        for key in self.condition_global_keys_list:
            nn.init.normal_(getattr(self, f"proj_{key}").mlp[0].weight, std=0.02)
            nn.init.normal_(getattr(self, f"proj_{key}").mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, condition_params, **kwargs):
        """
        Forward pass of SiT.
        x: (N, T, D) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        condition_params: dictionary of conditioning parameters
        """
        x = self.input_proj(x)                  # (N, T, D)
        t = self.t_embedder(t)                   # (N, D)

        condition_global = condition_params["condition_global"]
        c = t
        for idx, key in enumerate(self.condition_global_keys_list):
            y = getattr(self, f"proj_{key}")(condition_global[idx].squeeze(-1))
            c = c + y     

        for idx, block in enumerate(self.blocks):
            if self.is_pe_latent:
                pe_raw_dict = condition_params["pe_latent"]
                pe = self.pe_embed_list[idx](**pe_raw_dict)
                x = x + pe
            elif self.is_pecode_latent:
                pe = self.pe_embed_list[idx](condition_params["pe_latent"]["code"])
                if self.is_pecode_latent_pe:
                    pe = pe + self.pe_embed_list2[idx](**condition_params["pe_latent"]["raw_pe"])
                x = x + pe
            x = block(x, c)                      # (N, T, D)

        x = self.final_layer(x, c)                # (N, T, C)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=2)
        return {
            "x": x,
        }

    def get_latent_and_condition(self, batch, vae, is_drop_cond=False, is_mode=False, is_run_sqvae=True):
        latent_dict = vae.encode(batch, is_mode=is_mode, is_run_sqvae=is_run_sqvae)
        x = latent_dict["lat"] if is_run_sqvae else None
        if is_drop_cond:
            condition_global = [torch.ones_like(batch[k]) * -1 for k in self.condition_global_keys_list]
        else:
            condition_global = [batch[k] for k in self.condition_global_keys_list]
        condition = latent_dict["lat_cond"]
        if self.is_geom_last_feature:
            if "cond_last_feature" in latent_dict:
                condition = latent_dict["cond_last_feature"]
            else:
                condition = vae.cond_vae.return_last_features_and_norm(condition)
        elif self.is_fps_cond:
            n = self.n_latents
            fps_input = batch[f"xyz_fps_{n}"]
            for k in ["normal"]:
                fps_input = torch.cat([fps_input, batch[f"{k}_fps_{n}"]], dim=-1)
            condition = fps_input
        pe_latent = None
        if self.is_pe_latent:
            n = self.n_latents
            fps_input = batch[f"xyz_fps_{n}"]
            fps_context_dict = {}
            for k in ["normal"]:
                fps_context_dict[k] = batch[f"{k}_fps_{n}"]
            pe_latent = {"input": fps_input, "context_dict": fps_context_dict}
        elif self.is_pecode_latent:
            pe_latent = {
                "code": condition,
            }
            if self.is_pecode_latent_pe:
                n = self.n_latents
                fps_input = batch[f"xyz_fps_{n}"]
                fps_context_dict = {}
                for k in ["normal"]:
                    fps_context_dict[k] = batch[f"{k}_fps_{n}"]
                pe_latent["raw_pe"] =  {"input": fps_input, "context_dict": fps_context_dict}
            condition = None

        return {
            "latent": x,
            "latent_dict": latent_dict,
            "condition_params": {
                "condition": condition,
                "condition_global": condition_global,
                "pe_latent": pe_latent,
            }
        }

class SiTLoss:

    def __call__(self, net, inputs, condition_params=None, generator=None, noise_l=0, noise_r=1, args=None, ae=None, batch=None, latent_dict=None):
        transport = args.sit_transport
        model_kwargs = {
            "condition_params": condition_params,
        }
        model_without_ddp = net.module if hasattr(net, "module") else net

        loss_dict = transport.training_losses(net, model_without_ddp.normalize_latents(inputs), model_kwargs)

        ret = {
            "loss": loss_dict["loss"].mean(),
        }

        return ret

def create_sqdiffuse_from_config(config, vae_config):
    print("sqdiffuse_configs", config)
    print("vae_configs", vae_config)

    is_last_feature = config.is_last_feature if hasattr(config, "is_last_feature") else 0
    is_fps_cond = config.is_fps_cond if hasattr(config, "is_fps_cond") else 0
    
    if is_last_feature:
        in_channels_cond = vae_config.cond_vae_params.dim
    elif is_fps_cond:
        in_channels_cond = 6 # xyz, normal
    else:
        in_channels_cond = vae_config.cond_vae_params.latent_dim

    qk_norm = config.qk_norm if hasattr(config, "qk_norm") else 0

    is_pe_latent = config.is_pe_latent if hasattr(config, "is_pe_latent") else 0
    is_pecode_latent = config.is_pecode_latent if hasattr(config, "is_pecode_latent") else 0
    is_pecode_latent_pe = config.is_pecode_latent_pe if hasattr(config, "is_pecode_latent_pe") else 0
    target_mu = list(config.target_mu) if hasattr(config, "target_mu") else None
    target_std = list(config.target_std) if hasattr(config, "target_std") else None
    assert target_mu is not None and target_std is not None, "SQDiffuse config must define target_mu and target_std"

    assert not (is_pe_latent and is_pecode_latent), "is_pe_latent and is_pecode_latent cannot be True at the same time"
    if is_pecode_latent_pe: assert is_pecode_latent, "is_pecode_latent_pe requires is_pecode_latent=True"
    model = SiT(
        n_latents=vae_config.latent_vae_params.num_latents,
        in_channels=vae_config.latent_vae_params.latent_dim,
        depth=config.depth,
        n_heads=config.n_heads,
        d_head=config.d_head,
        condition_global_keys_list=config.condition_global_keys_list,
        in_channels_cond=in_channels_cond,
        is_geom_last_feature=is_last_feature,
        is_fps_cond=is_fps_cond,
        qk_norm=qk_norm,
        is_pe_latent=is_pe_latent,
        is_pecode_latent=is_pecode_latent,
        is_pecode_latent_pe=is_pecode_latent_pe,
        target_mu=target_mu,
        target_std=target_std,
    )
    return model