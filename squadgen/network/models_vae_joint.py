import torch
from torch import nn

from .models_ae import create_vae_from_config

class KLAutoEncoderJoint(nn.Module):
    def __init__(
        self,
        latent_vae_params,
        cond_vae_params,
    ):
        super().__init__()

        if not hasattr(latent_vae_params, "use_geom_code_as_fps"):
            latent_vae_params.use_geom_code_as_fps = 0
        self.use_geom_code_as_fps = latent_vae_params.use_geom_code_as_fps

        self.cond_vae_params = cond_vae_params
        self.latent_vae_params = latent_vae_params
        self.cond_vae = create_vae_from_config(cond_vae_params)
        self.latent_vae = create_vae_from_config(latent_vae_params)
        self.freeze_cond_vae = False

    def encode(self, batch, is_mode=False, is_latent_mode=False, is_cond_mode=False, only_trainable=False, is_run_sqvae=True):
        if self.use_geom_code_as_fps == 1:
                # forward geom code, last feature, no grad
                with torch.no_grad():
                    cond_enc = self.cond_vae.encode(batch, is_mode=is_mode or is_cond_mode)
                    cond_enc["last_feature"] = self.cond_vae.return_last_features_and_norm(cond_enc["latent"])
        elif self.freeze_cond_vae:
            if not only_trainable:
                with torch.no_grad():
                    cond_enc = self.cond_vae.encode(batch, is_mode=is_mode or is_cond_mode)
            else:
                cond_enc = {
                    "input": None,
                    "kl": None,
                    "latent": None,
                    "posterior": None,
                }
        else:
            # train cond_vae
            cond_enc = self.cond_vae.encode(batch, is_mode=is_mode or is_cond_mode)
        lat_cond = cond_enc["latent"]

        ans = {
            "lat": None,
            "lat_cond": lat_cond,
            "lat_end": None,
            "kl": None,
            "kl_cond": cond_enc["kl"],
            "input": None,
            "input_cond": cond_enc["input"],
            "cond_enc": cond_enc,
            "lat_enc": None,
        }
        if "last_feature" in cond_enc:
            ans["cond_last_feature"] = cond_enc["last_feature"]
        if self.latent_vae is not None and is_run_sqvae:
            lat_enc = self.latent_vae.encode(batch, is_mode=is_mode or is_latent_mode, geom_code_fps=cond_enc["last_feature"] if self.use_geom_code_as_fps == 1 else None)
            lat = lat_enc["latent"]

            ans.update({
                "lat": lat,
                "kl": lat_enc["kl"],
                "input": lat_enc["input"],
                "lat_end": lat,
                "lat_enc": lat_enc,
            })
        return ans

    def decode(self, batch, latent, only_trainable=False):
        lat = latent["lat_end"]
        lat_cond = latent["lat_cond"]
        if self.freeze_cond_vae:
            if not only_trainable:
                with torch.no_grad():
                    out_cond = self.cond_vae.decode_batch(lat_cond, batch)
            else:
                out_cond = {}
        else:
            out_cond = self.cond_vae.decode_batch(lat_cond, batch)
        out = self.latent_vae.decode_batch(lat, batch) if self.latent_vae is not None else out_cond
        return {
            "out": out,
            "out_cond": out_cond,
        }

    def get_label(self, batch):
        ans = {
            "gt": {},
            "gt_cond": {k: batch[k] if k.endswith("_near") or k.endswith("_global") else batch[f"{k}_query"] for k in self.cond_vae.out_keys_list},
        }
        if self.latent_vae is not None:
            ans.update({
                "gt": {k: batch[k.replace("_edge", "_queryedge")] if k.endswith("_edge") else batch[f"{k}_query"] for k in self.latent_vae.out_keys_list},
            })
        return ans

    def forward(self, batch, is_mode=False, is_latent_mode=False, is_cond_mode=False, decode=True):
        latent = self.encode(batch, is_mode=is_mode, is_latent_mode=is_latent_mode, is_cond_mode=is_cond_mode, only_trainable=True)
        out = None
        if decode:
            out = self.decode(batch, latent, only_trainable=True)
        return {
            "out": out,
            "latent": latent,
        }

def create_joint_vae_from_config(config):
    model = KLAutoEncoderJoint(
        latent_vae_params=config.latent_vae_params,
        cond_vae_params=config.cond_vae_params,
    )
    return model
