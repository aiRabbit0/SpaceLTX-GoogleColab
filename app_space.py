from __future__ import annotations

import asyncio
import glob
import json
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid
from typing import Any

import base64
import threading
import time

import gradio as gr
import requests as http_requests
import spaces
import torch
from huggingface_hub import hf_hub_download
from PIL import Image


ROOT = pathlib.Path(__file__).resolve().parent
COMFY = ROOT / "ComfyUI"
MODELS = COMFY / "models"
INPUT = COMFY / "input"
OUTPUT = COMFY / "output"

WORKFLOW_REPO = "TenStrip/LTX2.3-10Eros_Workflows"
WORKFLOW_REVISION = "1b8e8988842a5850dbba58d732c3e29ce430c1c7"
WORKFLOW_FILENAME = "10Eros_10SNodes_LikenessGuideHelper_I2V_v3.2.json"

# Bundled multi-reference workflow shipped alongside app.py. Used when the
# "multi-reference (original)" input_mode is selected. Patched at conversion
# time to use our checkpoint instead of the split UNET/VAE/CLIP loader chain
# the workflow ships with.
RUNEXX_WORKFLOW_FILE = "runexx_msr_workflow.json"

# Visual-form node ids in the bundled runexx workflow. Used during
# conversion to patch node types/widgets, set up rewires, and inject
# user inputs (prompt, images, seed, dimensions).
RUNEXX_NODE_UNET_LOADER = 59          # UNETLoader -> CheckpointLoaderSimple
RUNEXX_NODE_CLIP_LOADER = 57          # DualCLIPLoader -> LTXAVTextEncoderLoader
RUNEXX_NODE_VAE_VIDEO = 56            # VAELoader (video) -> use checkpoint vae
RUNEXX_NODE_VAE_AUDIO = 53            # VAELoaderKJ -> LTXVAudioVAELoader
RUNEXX_NODE_VAE_TINY = 55             # VAELoader (preview) -> skip
RUNEXX_NODE_DISTILLED_LORA = 60       # LoraLoaderModelOnly -> skip
RUNEXX_NODE_GGUF_UNET = 1257          # UnetLoaderGGUF -> skip (parallel path)
RUNEXX_NODE_GGUF_CLIP = 1256          # DualCLIPLoaderGGUF -> skip
RUNEXX_NODE_UUID_IMAGESIZE = 1222     # unknown UUID with 4 INT outputs (w/h)
RUNEXX_NODE_UUID_CONDITIONING = 1245  # unknown UUID feeding pass-1 CropGuides
RUNEXX_NODE_SAMPLER_SWITCH = 1235     # ComfySwitchNode toggling pass-1/pass-2
# IC-LoRA + MSR architectural nodes (we PRESERVE these intact)
RUNEXX_NODE_LICON_MSR = 28            # LiconMSR
RUNEXX_NODE_ICLORA_GUIDE_P1 = 9       # LTXAddVideoICLoRAGuide pass 1
RUNEXX_NODE_ICLORA_GUIDE_P2 = 1229    # LTXAddVideoICLoRAGuide pass 2
RUNEXX_NODE_CROP_GUIDES_P1 = 17       # LTXVCropGuides pass 1
RUNEXX_NODE_CROP_GUIDES_P2 = 132      # LTXVCropGuides pass 2
RUNEXX_NODE_SAMPLER_P1 = 16           # SamplerCustomAdvanced pass 1
RUNEXX_NODE_SAMPLER_P2 = 133          # SamplerCustomAdvanced pass 2
# User-input mapping
RUNEXX_NODE_LOAD_IMAGE_REF1 = 33      # main reference image
RUNEXX_NODE_LOAD_IMAGE_REF2 = 29      # second reference image
RUNEXX_NODE_LOAD_IMAGE_BG = 30        # background reference image
RUNEXX_NODE_CLIPTEXT_POS = 5          # positive prompt
RUNEXX_NODE_CLIPTEXT_NEG = 6          # negative prompt
RUNEXX_NODE_RANDOM_NOISE = 15         # seed
RUNEXX_NODE_WIDTH_CONST = 166         # INTConstant width
RUNEXX_NODE_HEIGHT_CONST = 167        # INTConstant height
RUNEXX_NODE_EMPTY_LATENT = 8          # EmptyLTXVLatentVideo

CUSTOM_NODES = [
    ("ComfyUI-GGUF", "https://github.com/city96/ComfyUI-GGUF.git"),
    ("ComfyUI-LTXVideo", "https://github.com/Lightricks/ComfyUI-LTXVideo.git"),
    ("10S-Comfy-nodes", "https://github.com/TenStrip/10S-Comfy-nodes.git"),
    ("ComfyUI-KJNodes", "https://github.com/kijai/ComfyUI-KJNodes.git"),
    ("rgthree-comfy", "https://github.com/rgthree/rgthree-comfy.git"),
    ("ComfyUI-VideoHelperSuite", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"),
    ("RES4LYF", "https://github.com/ClownsharkBatwing/RES4LYF.git"),
    ("ComfyUI-Easy-Use", "https://github.com/yolain/ComfyUI-Easy-Use.git"),
    ("ComfyUI-mxToolkit", "https://github.com/Smirnov75/ComfyUI-mxToolkit.git"),
    ("ComfyMath", "https://github.com/evanspearman/ComfyMath.git"),
    ("ComfyUI-Licon-MSR", "https://github.com/liconstudio/ComfyUI-Licon-MSR.git"),
    ("ComfyUI-RMBG", "https://github.com/1038lab/ComfyUI-RMBG.git"),
    ("ComfyUI-PromptRelay", "https://github.com/kijai/ComfyUI-PromptRelay.git"),
    ("ComfyUI-FunPack", "https://github.com/digital-garbage/ComfyUI-FunPack.git"),
    ("ComfyUI-MelBandRoFormer", "https://github.com/kijai/ComfyUI-MelBandRoFormer.git"),
    ("ComfyUI-MultiLoRALoader", "https://github.com/phazei/ComfyUI-MultiLoRALoader.git"),
]

# Local wrapper nodes, written into comfy's custom_nodes at startup.
_KV_WRAPPER_CODE = '''import sys, pathlib, traceback
import torch


_kv_strength_scale = [1.0]


def _av_patch_extend_v_pe(module):
    """LTX-AV compat for funpack. Idempotent.
    - _extend_v_pe also extends video CompressedTimestep modulation tensors
      + v_cross_pe (a2v cross-attn). Without this, AV crashes at:
        av_model.py:274 (vscale_msa size mismatch) -> timestep extension
        av_model.py:322 (audio_to_video_attn rope dim mismatch) -> v_cross_pe
        (apply_split_rotary_emb's reshape branch needs T=T_q)
    - _sigma_gated_strength multiplies base_strength by _kv_strength_scale so
      the wrapper's strength input scales every K/V hook firing."""
    if getattr(module, "_av_patched", False):
        return
    orig_extend = module._extend_v_pe
    orig_gated = module._sigma_gated_strength
    av_timestep_keys = (
        "v_timestep",
        "v_cross_scale_shift_timestep",
        "v_cross_gate_timestep",
        "v_prompt_timestep",
    )

    def _extend_pe_entry(pe, n_ref):
        """Extend a freqs_cis tuple (cos, sin[, split_mode]) by prepending
        n_ref neutral-rotation entries (cos=1, sin=0)."""
        try:
            cos, sin = pe[0], pe[1]
            dev, dt = cos.device, cos.dtype
            ndim = cos.ndim
            if ndim == 4:
                r = (cos.shape[0], cos.shape[1], n_ref, cos.shape[3])
                dim = 2
            elif ndim == 3:
                r = (cos.shape[0], n_ref, cos.shape[2])
                dim = 1
            elif ndim == 2:
                r = (n_ref, cos.shape[1])
                dim = 0
            else:
                return pe
            ref_cos = torch.ones(r, device=dev, dtype=dt)
            ref_sin = torch.zeros(r, device=dev, dtype=dt)
            ext_cos = torch.cat([ref_cos, cos], dim=dim)
            ext_sin = torch.cat([ref_sin, sin], dim=dim)
            tail = tuple(pe[2:]) if len(pe) > 2 else ()
            return (ext_cos, ext_sin) + tail
        except Exception:
            return pe

    _prefix_cls_cache = {}
    # Reused zero-prefix tensors keyed by shape. Without this we'd allocate
    # ~36MB per ada-param per block per step; the resulting churn fragments
    # the allocator and surfaces as NVML asserts in the subsequent VAE decode.
    _zero_prefix_cache = {}

    def _get_zero_prefix(n_ref, batch_size, dim, device, dtype):
        key = (n_ref, batch_size, dim, str(device), dtype)
        z = _zero_prefix_cache.get(key)
        if z is None:
            z = torch.zeros(batch_size, n_ref, dim, device=device, dtype=dtype)
            _zero_prefix_cache[key] = z
        return z

    def _make_prefix_subclass(base_cls):
        cached = _prefix_cls_cache.get(base_cls)
        if cached is not None:
            return cached

        class _RefPrefixedTimestep(base_cls):
            __slots__ = ("_n_ref",)

            def __init__(self, base, n_ref):
                # Bypass parent __init__ (which expects raw tensor + ppf);
                # mirror attributes from the base instance and share data.
                self.batch_size = base.batch_size
                self.num_frames = base.num_frames
                self.patches_per_frame = base.patches_per_frame
                self.feature_dim = base.feature_dim
                self.data = base.data
                self._n_ref = int(n_ref)

            def expand(self):
                original = super().expand()
                if self._n_ref == 0:
                    return original
                zeros = _get_zero_prefix(
                    self._n_ref, original.shape[0], original.shape[2],
                    original.device, original.dtype,
                )
                return torch.cat([zeros, original], dim=1)

            def expand_for_computation(self, scale_shift_table, batch_size,
                                       indices=slice(None, None)):
                original = super().expand_for_computation(
                    scale_shift_table, batch_size, indices
                )
                if self._n_ref == 0:
                    return original
                prefixed = []
                for t in original:
                    zeros = _get_zero_prefix(
                        self._n_ref, t.shape[0], t.shape[2],
                        t.device, t.dtype,
                    )
                    prefixed.append(torch.cat([zeros, t], dim=1))
                return tuple(prefixed)

        _prefix_cls_cache[base_cls] = _RefPrefixedTimestep
        return _RefPrefixedTimestep

    def _extend_av(kwargs, n_ref):
        new_kwargs = orig_extend(kwargs, n_ref)
        n_ref_int = int(n_ref)
        for key in av_timestep_keys:
            ts = new_kwargs.get(key)
            if ts is None:
                continue
            # CompressedTimestep duck-typing
            if not (hasattr(ts, "data") and hasattr(ts, "patches_per_frame")
                    and hasattr(ts, "num_frames")):
                continue
            try:
                ppf = max(1, int(getattr(ts, "patches_per_frame", 1) or 1))
                if ppf == 1 or n_ref_int % ppf == 0:
                    # Aligned: extend compressed storage in-place.
                    ref_frames = n_ref_int if ppf == 1 else n_ref_int // ppf
                    data = ts.data
                    ref_data = torch.zeros(
                        data.shape[0],
                        ref_frames,
                        data.shape[2],
                        device=data.device,
                        dtype=data.dtype,
                    )
                    new_data = torch.cat([ref_data, data], dim=1)
                    new_ts = type(ts).__new__(type(ts))
                    new_ts.data = new_data
                    new_ts.batch_size = ts.batch_size
                    new_ts.num_frames = ref_frames + ts.num_frames
                    new_ts.patches_per_frame = ts.patches_per_frame
                    new_ts.feature_dim = ts.feature_dim
                else:
                    # Misaligned (e.g. pass-2 tile sampler ppf doesn't divide
                    # pass-1 n_ref): wrap so storage stays compressed.
                    PrefixCls = _make_prefix_subclass(type(ts))
                    new_ts = PrefixCls(ts, n_ref_int)
                new_kwargs = dict(new_kwargs)
                new_kwargs[key] = new_ts
            except Exception as e:
                print(f"[FunPackKVApply] could not extend {key}: {e}", flush=True)
        v_cross_pe = new_kwargs.get("v_cross_pe")
        if v_cross_pe is not None:
            try:
                ext_pe = _extend_pe_entry(v_cross_pe, n_ref)
                if ext_pe is not v_cross_pe:
                    new_kwargs = dict(new_kwargs)
                    new_kwargs["v_cross_pe"] = ext_pe
            except Exception as e:
                print(f"[FunPackKVApply] could not extend v_cross_pe: {e}", flush=True)
        return new_kwargs

    def _gated_scaled(base_strength, sigma, sigma_high, sigma_low):
        # Scale base_strength by user knob, then delegate to funpack's ramp.
        return orig_gated(
            base_strength * _kv_strength_scale[0], sigma, sigma_high, sigma_low,
        )

    module._extend_v_pe = _extend_av
    module._sigma_gated_strength = _gated_scaled
    module._av_patched = True


class FunPackKVApply:
    """Minimal wrapper for funpack's build_enhancements. Calls it with stub
    rating_profile/refinement_key/reward so only the K/V in-context path
    fires; AV compatibility patches applied via _av_patch_extend_v_pe."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "conditioning": ("CONDITIONING",),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
            },
            "optional": {
                "temporal_style": (
                    ["natural", "accelerate", "decelerate", "loop", "freeze"],
                    {"default": "natural"},
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "conditioning")
    FUNCTION = "apply"
    CATEGORY = "FunPack/Wrapper"

    def apply(self, model, latent, conditioning, strength=1.0, temporal_style="natural"):
        try:
            funpack_dir = None
            this_dir = pathlib.Path(__file__).resolve().parent
            for parent in [this_dir.parent] + list(this_dir.parent.parents)[:3]:
                for name in ("ComfyUI-FunPack", "ComfyUI_FunPack"):
                    candidate = parent / name
                    if (candidate / "ltx_enhancements.py").exists():
                        funpack_dir = str(candidate)
                        break
                if funpack_dir:
                    break

            if funpack_dir and funpack_dir not in sys.path:
                sys.path.insert(0, funpack_dir)

            try:
                import ltx_enhancements
                build_enhancements = ltx_enhancements.build_enhancements
            except ImportError as exc:
                print(f"[FunPackKVApply] could not import build_enhancements: {exc}", flush=True)
                return (model, conditioning)

            # Install AV compat + strength-scaling monkey-patches, then push
            # the user knob into the module-level scale before build runs.
            _av_patch_extend_v_pe(ltx_enhancements)
            _kv_strength_scale[0] = float(strength)

            patched = build_enhancements(
                model,
                rating_profile={},
                temporal_style=temporal_style,
                refinement_key="",
                reward=0.0,
                reference_latent=latent,
                conditioning=conditioning,
            )
            return (patched, conditioning)
        except Exception as exc:
            print(f"[FunPackKVApply] failed: {exc}", flush=True)
            traceback.print_exc()
            return (model, conditioning)


class AudioRefPrep:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "normalize": ("BOOLEAN", {"default": True}),
                "max_seconds": ("FLOAT", {
                    "default": 10.0, "min": 1.0, "max": 60.0, "step": 0.5,
                }),
                "target_peak_db": ("FLOAT", {
                    "default": -3.0, "min": -24.0, "max": 0.0, "step": 0.5,
                }),
                "max_gain_db": ("FLOAT", {
                    "default": 24.0, "min": 0.0, "max": 60.0, "step": 1.0,
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process"
    CATEGORY = "audio"

    def process(self, audio, normalize=True, max_seconds=10.0,
                target_peak_db=-3.0, max_gain_db=24.0):
        try:
            waveform = audio.get("waveform")
            sample_rate = int(audio.get("sample_rate", 44100))
            if waveform is None:
                return (audio,)

            out = waveform.detach().clone()
            max_samples = int(max(1.0, float(max_seconds)) * sample_rate)
            if max_samples > 0 and out.shape[-1] > max_samples:
                out = out[..., :max_samples]

            if normalize:
                peak = out.abs().amax()
                peak_value = float(peak.detach().cpu())
                if bool(torch.isfinite(peak).item()) and peak_value > 1e-8:
                    target_peak = 10 ** (float(target_peak_db) / 20.0)
                    max_gain = 10 ** (float(max_gain_db) / 20.0)
                    gain = min(target_peak / peak_value, max_gain)
                    out = (out * gain).clamp(-1.0, 1.0)

            return ({"waveform": out.contiguous(), "sample_rate": sample_rate},)
        except Exception as exc:
            print(f"[AudioRefPrep] failed: {exc}", flush=True)
            traceback.print_exc()
            return (audio,)


NODE_CLASS_MAPPINGS = {
    "FunPackKVApply": FunPackKVApply,
    "AudioRefPrep": AudioRefPrep,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FunPackKVApply": "FunPack KV Apply",
    "AudioRefPrep": "Audio Ref Prep",
}
'''


def _install_kv_wrapper(comfy_root: pathlib.Path) -> None:
    """Write the FunPackKVApply wrapper file into comfy's custom_nodes so
    it gets loaded with the other custom nodes. Idempotent."""
    target_dir = comfy_root / "custom_nodes" / "funpack_kv_apply"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "__init__.py"
    if target_file.exists() and target_file.read_text(encoding="utf-8") == _KV_WRAPPER_CODE:
        return
    target_file.write_text(_KV_WRAPPER_CODE, encoding="utf-8")

DOWNLOADS = [
    {
        "repo": "TenStrip/LTX2.3-10Eros",
        "file": "10Eros_v1-fp8mixed_learned.safetensors",
        "dest": MODELS / "checkpoints" / "10Eros_v1-fp8mixed_learned.safetensors",
        "label": "main checkpoint",
    },
    {
        "repo": "Comfy-Org/ltx-2",
        "file": "split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors",
        "dest": MODELS / "text_encoders" / "gemma_3_12B_it_fp8_scaled.safetensors",
        "label": "text encoder",
    },
    {
        "repo": "TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments",
        "file": "ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors",
        "dest": MODELS / "loras" / "ltx23" / "ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors",
        "label": "distilled lora",
    },
    {
        "repo": "VasiliyWeb/OmniNFT_ComfyUI",
        "file": "OmniNFT_converted_lora.safetensors",
        "dest": MODELS / "loras" / "ltx23" / "OmniNFT_converted_lora.safetensors",
        "label": "omninft (converted) lora",
    },
    {
        "repo": "Kijai/LTX2.3_comfy",
        "file": "loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",
        "dest": MODELS / "loras" / "ltx23" / "LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",
        "label": "omninft RL bf16 lora",
        "slot": "slot4",
    },
    {
        "repo": "Lightricks/LTX-2.3",
        "file": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "dest": MODELS / "latent_upscale_models" / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "label": "spatial upscaler",
    },
    {
        "repo": "maximsobolev275/LTX-SulphurExperimental-LoRA-Optimized",
        "file": "LTX_SulphurEXP_LoRA_fro99-avgrank105.safetensors",
        "dest": MODELS / "loras" / "ltx23" / "LTX_SulphurEXP_LoRA_fro99-avgrank105.safetensors",
        "label": "sulphur experimental lora",
    },
    {
        "repo": "SulphurAI/Sulphur-2-base",
        "file": "experimental/sulphur_experimental_lora_v1.safetensors",
        "dest": MODELS / "loras" / "ltx23" / "sulphur_experimental_lora_v1.safetensors",
        "label": "sulphur experimental v1 lora (kiwv official)",
    },
    {
        "repo": "signsur4739379373/archive",
        "file": "2497207_LTX2.3_reasoning_I2V_V3.safetensors",
        "dest": MODELS / "loras" / "ltx23" / "2497207_LTX2.3_reasoning_I2V_V3.safetensors",
        "label": "vbvr lora",
    },
    {
        "repo": "signsur4739379373/archive",
        "file": "1811313_dreamlay_ltx_V2.safetensors",
        "dest": MODELS / "loras" / "ltx23" / "1811313_dreamlay_ltx_V2.safetensors",
        "label": "dreamly lora",
    },
    {
        "repo": "signsur4739379373/archive",
        "file": "2509189_Synth_01_rank32.safetensors",
        "dest": MODELS / "loras" / "ltx23" / "2509189_Synth_01_rank32.safetensors",
        "label": "synth lora",
        "slot": "slot2",
    },
    {
        "repo": "signsur4739379373/archive",
        "file": "2598050_plora_sulfer_v1.2-step00008500.safetensors",
        "dest": MODELS / "loras" / "ltx23" / "2598050_plora_sulfer_v1.2-step00008500.safetensors",
        "label": "plora",
        "slot": "slot3",
    },
    {
        "repo": "signsur4739379373/archive",
        "file": "2344781_Sulphur_LTX 2.3_better_motion.safetensors",
        "dest": MODELS / "loras" / "ltx23" / "2344781_Sulphur_LTX 2.3_better_motion.safetensors",
        "label": "better motion lora (mistic)",
        "slot": "slot5",
    },
    {
        "repo": "signsur4739379373/archive",
        "file": "2592090_LTX2.3_Physics_V2_000002000.safetensors",
        "dest": MODELS / "loras" / "ltx23" / "2592090_LTX2.3_Physics_V2_000002000.safetensors",
        "label": "physics v2 lora (mistic)",
        "slot": "slot6",
    },
    {
        "repo": "signsur4739379373/archive",
        "file": "2508281_LTX-2.3_Cinematic hardcut.safetensors",
        "dest": MODELS / "loras" / "ltx23" / "2508281_LTX-2.3_Cinematic hardcut.safetensors",
        "label": "cinematic hardcut lora",
        "slot": "slot1",
    },
    {
        "repo": "joyfox/LTX-2.3-Transition-LORA",
        "file": "ltx2.3-transition.safetensors",
        "dest": MODELS / "loras" / "ltx23" / "ltx2.3-transition.safetensors",
        "label": "transition lora",
    },
    {
        "repo": "LiconStudio/LTX-2.3-Multiple-Subject-Reference",
        "file": "LTX2.3-Licon-MSR-test_version.safetensors",
        "dest": MODELS / "loras" / "ltx23" / "LTX2.3-Licon-MSR-test_version.safetensors",
        "label": "MSR ic-lora",
    },
    {
        "repo": "WarmBloodAban/Singularity-LTX-2.3_OmniCine_V1",
        "file": "Singularity-LTX-2.3_OmniCine_V1nsf.safetensors",
        "dest": MODELS / "loras" / "ltx23" / "Singularity-LTX-2.3_OmniCine_V1nsf.safetensors",
        "label": "singularity lora",
    },
    {
        "repo": "Kijai/MelBandRoFormer_comfy",
        "file": "MelBandRoformer_fp16.safetensors",
        "dest": MODELS / "diffusion_models" / "MelBandRoformer_fp16.safetensors",
        "label": "mel band roformer (stem separation)",
    },
]

SULPHUR_LORA_FILENAME = "ltx23/LTX_SulphurEXP_LoRA_fro99-avgrank105.safetensors"
SULPHUR_V1_LORA_FILENAME = "ltx23/sulphur_experimental_lora_v1.safetensors"
VBVR_LORA_FILENAME = "ltx23/2497207_LTX2.3_reasoning_I2V_V3.safetensors"
DREAMLY_LORA_FILENAME = "ltx23/1811313_dreamlay_ltx_V2.safetensors"
SYNTH_LORA_FILENAME = "ltx23/2509189_Synth_01_rank32.safetensors"
PLORA_LORA_FILENAME = "ltx23/2598050_plora_sulfer_v1.2-step00008500.safetensors"
BETTER_MOTION_LORA_FILENAME = "ltx23/2344781_Sulphur_LTX 2.3_better_motion.safetensors"
PHYSICS_V2_LORA_FILENAME = "ltx23/2592090_LTX2.3_Physics_V2_000002000.safetensors"
SINGULARITY_LORA_FILENAME = "ltx23/Singularity-LTX-2.3_OmniCine_V1nsf.safetensors"
OMNINFT_LORA_FILENAME = "ltx23/OmniNFT_converted_lora.safetensors"
OMNINFT_BF16_LORA_FILENAME = "ltx23/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors"
MSR_LORA_FILENAME = "ltx23/LTX2.3-Licon-MSR-test_version.safetensors"
HARDCUT_LORA_FILENAME = "ltx23/2508281_LTX-2.3_Cinematic hardcut.safetensors"
TRANSITION_LORA_FILENAME = "ltx23/ltx2.3-transition.safetensors"
# Runtime lora-file selection for the 7 swappable slots. Each slot gets a UI
# dropdown listing its original file plus every .safetensors installed under
# loras/ltx23/custom/ (populated externally, e.g. by the Colab notebook before
# the UI is built). manifest.json there maps filename -> CivitAI metadata for
# friendlier display names. Slot 7 has no original lora: it defaults to
# LORA_NONE and only does anything once a custom file is selected.
LORA_NONE = "(none)"

# Comma-separated slot keys (e.g. "slot1,slot4") whose ORIGINAL lora should not
# be downloaded nor offered in the dropdown — set by the Colab notebook's
# per-lora checkboxes to save disk. The slot itself stays usable with customs.
_SKIP_SLOT_LORAS = {
    s.strip() for s in os.environ.get("SKIP_SLOT_LORAS", "").split(",") if s.strip()
}


def _slot_original(slot_key: str, filename: str) -> str | None:
    """The slot's original lora file, or None when its download is skipped."""
    return None if slot_key in _SKIP_SLOT_LORAS else filename


def _slot_lora_choices(original: str | None) -> list:
    choices = []
    if original:
        choices.append((f"original: {pathlib.Path(original).name}", original))
    custom_dir = MODELS / "loras" / "ltx23" / "custom"
    manifest: dict = {}
    manifest_path = custom_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    if custom_dir.exists():
        for f in sorted(custom_dir.glob("*.safetensors")):
            display = (manifest.get(f.name) or {}).get("name") or f.stem
            choices.append((f"custom: {str(display)[:40]}", f"ltx23/custom/{f.name}"))
    choices.append((LORA_NONE, LORA_NONE))
    return choices
NODE_POWER_LORA = "557"

# Workflow has two sampler passes; MSR conditioning injected at pass-1
# start (feeds both passes via shared positive/negative chain), trailing
# conditioning frames cropped at pass-2 end before final VAE decode.
# - 806 LikenessGuide / 827 LikenessAnchor / 731 LatentAnchorAware: bypassed.
# - 772 LTXVImgToVideoInplaceKJ (pass 1): 548 ConcatAVLatent rewired through MSR guide.
# - 596 LTXVSeparateAVLatent (pass 2 / final): video output rewired through CropGuides.
# - 740 VAEDecode (pass 2 / final): samples rewired to CropGuides output.
# Pass-1 separator 556 + pass-1 decoder 552 are excluded from API workflow
# via skip_ids so they are NOT valid crop/decode targets.
MSR_NODE_LIKENESS_GUIDE = "806"
MSR_NODE_LIKENESS_ANCHOR = "827"
MSR_NODE_LATENT_ANCHOR = "731"
MSR_NODE_INPLACE_PASS1 = "772"
MSR_NODE_CONCAT_PASS1 = "548"
MSR_NODE_FINAL_SEPARATE = "596"
MSR_NODE_VAE_DECODE = "740"
# Source-of-truth latent length node. Its `length` widget is overridden when
# MSR is on to add headroom for the pseudo-video frames that
# LTXAddVideoICLoRAGuide consumes (the IC-LoRA asserts conditioning frames
# fit within latent_length).
MSR_NODE_EMPTY_LATENT = "534"

# IDs added by the MSR injection, prefix-namespaced to avoid collision with
# numeric ids of the imported visual workflow.
MSR_NEW_PSEUDO_VIDEO = "msr_pseudo"
MSR_NEW_GUIDE = "msr_guide"
MSR_NEW_GUIDE_MULTI = "msr_guide_multi"
MSR_NEW_CROP = "msr_crop"
MSR_NEW_REF_2 = "msr_ref_2"
MSR_NEW_REF_3 = "msr_ref_3"
MSR_NEW_REF_4 = "msr_ref_4"
MSR_NEW_BG = "msr_bg"
# LTXICLoRALoaderModelOnly node: installs IC-LoRA-specific model hooks +
# extracts reference_downscale_factor from safetensors metadata. Plain
# Power Lora Loader only loads weights without these hooks.
MSR_NEW_ICLORA_LOADER = "msr_iclora_loader"

# Prompt Relay injection (timeline-based text conditioning).
# Adds a single PromptRelaySmartEncode node spliced between Power Lora Loader
# and its downstream LTX2LoraLoaderAdvanced consumers. The node patches
# the model (attention prior) AND outputs new positive conditioning.
# Disabled when MSR is on (model chain is already rewired by MSR injection).
RELAY_NEW_NODE = "prompt_relay"
NODE_TEXT_ENCODER = "616"   # LTXAVTextEncoderLoader, provides CLIP
NODE_LTXV_CONDITIONING = "523"  # consumes positive from CLIPTextEncode 536

# FunPack scene chain injection. Replaces the first-pass sampler with
# FunPackLTXAVSceneChainSampler and routes its stitched latent directly into
# the final split/decode path (bypassing the pass-2 tiled sampler for v1).
SCENE_CHAIN_NEW_NODE = "scene_chain_sampler"
SCENE_CHAIN_NODE_PREFIX = "scene_chain"
NODE_FIRST_PASS_SAMPLER = "510"
NODE_FIRST_PASS_SAMPLER_SELECT = "520"
NODE_FIRST_PASS_SIGMAS = "652"
NODE_FIRST_PASS_LATENT = "548"
NODE_VIDEO_VAE = "559"
NODE_FINAL_SEPARATE = "596"

# K/V conditioning (FunPack ltx_enhancements.build_enhancements via wrapper).
# Splices a FunPackKVApply node between Power Lora Loader (557) and its
# downstream model consumers. Reads the i2v reference latent from
# LTXVImgToVideoInplaceKJ pass 1 (node 772) slot 0. Disabled when MSR
# mode is on (model chain already rewired).
KV_NEW_NODE = "kv_apply"
NODE_AUDIO_VAE_LOADER = "617"
AUDIO_REF_NEW_LOAD = "audio_ref_load"
AUDIO_REF_NEW_TRIM = "audio_ref_trim"
AUDIO_REF_NEW_MEL_LOADER = "audio_ref_mel_loader"
AUDIO_REF_NEW_MEL_SAMPLER = "audio_ref_mel_sampler"
AUDIO_REF_NEW_PREP = "audio_ref_prep"
AUDIO_REF_NEW_NODE = "audio_ref"
NODE_I2V_REF_LATENT = "772"  # LTXVImgToVideoInplaceKJ pass 1, slot 0

NODE_OUTPUT = "597"
NODE_LOAD_IMAGE = "834"
NODE_POSITIVE = "536"
NODE_NEGATIVE = "537"
NODE_SEED = "524"
NODE_WIDTH = "791"
NODE_HEIGHT = "792"
NODE_LENGTH = "796"
NODE_FIRST_FRAME = "797"
NODE_LIKENESS_GUIDE = "806"
NODE_LIKENESS_ANCHOR = "827"
NODE_LATENT_ANCHOR = "731"
NODE_REFINE_SIGMAS = "582"
PRESETS = ["original", "tuned", "tuned #2", "experimental #1"]

# Unified preset values. Each preset defines all user-facing params at once.
# Loras not listed in original TenStrip workflow default to 0.
_SIGMA_ORIGINAL = "0.715, 0.4824, 0.2412, 0.0"
_SIGMA_TUNED    = "0.4824, 0.2412, 0.0"

PRESET_VALUES = {
    "original": {
        # original TenStrip workflow values
        "mode": "anchor only",
        "sulphur_fro99": 0.0, "sulphur_v1": 0.0, "vbvr": 0.0,
        "dreamly": 0.0, "synth": 0.0, "plora": 0.0,
        "singularity": 0.0, "omninft": 0.8, "omninft_bf16": 0.0,
        "better_motion": 0.0, "physics_v2": 0.0, "hardcut": 0.0, "transition": 0.15,
        "likeness_strength": 0.9,
        "likeness_anchor_strength": 0.5,
        "latent_anchor_strength": 0.11,
        "first_frame_strength": 0.77,
        "anchor_similarity_threshold": 0.5,
        "energy_threshold": 0.3,
        "cache_warmup": 50,
        "sigma_string": _SIGMA_ORIGINAL,
    },
    "tuned": {
        "mode": "anchor only",
        "sulphur_fro99": 0.15, "sulphur_v1": 0.15, "vbvr": 0.5,
        "dreamly": 0.6, "synth": 0.0, "plora": 0.0,
        "singularity": 0.3, "omninft": 0.8, "omninft_bf16": 0.0,
        "better_motion": 0.0, "physics_v2": 0.0, "hardcut": 0.0, "transition": 0.15,
        "likeness_strength": 0.9,
        "likeness_anchor_strength": 0.15,
        "latent_anchor_strength": 0.08,
        "first_frame_strength": 0.82,
        "anchor_similarity_threshold": 0.3,
        "energy_threshold": 0.3,
        "cache_warmup": 400,
        "sigma_string": _SIGMA_TUNED,
    },
    "tuned #2": {
        "mode": "anchor only",
        "sulphur_fro99": 0.15, "sulphur_v1": 0.15, "vbvr": 0.5,
        "dreamly": 0.6, "synth": 0.0, "plora": 0.0,
        "singularity": 0.3, "omninft": 0.3, "omninft_bf16": 0.0,
        "better_motion": 0.0, "physics_v2": 0.0, "hardcut": 0.0, "transition": 0.15,
        "likeness_strength": 0.9,
        "likeness_anchor_strength": 0.15,
        "latent_anchor_strength": 0.08,
        "first_frame_strength": 0.82,
        "anchor_similarity_threshold": 0.3,
        "energy_threshold": 0.3,
        "cache_warmup": 400,
        "sigma_string": _SIGMA_TUNED,
    },
    "experimental #1": {
        # campaign #1 ideal settings (sobol parameter hunt results)
        "mode": "anchor only",
        "sulphur_fro99": 0.25, "sulphur_v1": 0.20, "vbvr": 0.85,
        "dreamly": 0.45, "synth": 0.30, "plora": 0.70,
        "singularity": 0.70, "omninft": 1.25, "omninft_bf16": 1.70,
        "better_motion": 0.30, "physics_v2": 0.70, "hardcut": 0.0, "transition": 0.15,
        "likeness_strength": 0.35,
        "likeness_anchor_strength": 0.72,
        "latent_anchor_strength": 0.33,
        "first_frame_strength": 0.67,
        "anchor_similarity_threshold": 0.65,
        "energy_threshold": 0.55,
        "cache_warmup": 400,
        "sigma_string": _SIGMA_TUNED,
    },
}

# Audio chain node ids kept by the converter so the native AV
# concat/separate/decoder nodes feed 597.audio properly. Node 789
# (TwoWaySwitch) is dropped (requires controlaltai-nodes not installed);
# its selected input (556 slot 1) is wired directly to 591.audio_latent
# via AUDIO_BYPASS_REWIRES.
AUDIO_CHAIN_NODE_IDS = {274, 535, 548, 550, 556, 591, 593, 596, 617}
# Silent-only sampler/decoder rewires dropped so the original AV
# concat/separate links survive conversion.
AUDIO_ONLY_REWIRE_KEYS = {"510", "744", "802", "740"}
# Bypass node 789 (TwoWaySwitch) by wiring 556 slot 1 directly into
# 591.audio_latent.
AUDIO_BYPASS_REWIRES = {
    "591": {"audio_latent": ["556", 1]},
}

DEFAULT_NEGATIVE = (
    "captions, music, transition, VR, bad quality, subtitles, text, watermark, "
    "overlay effects, cartoon, childish, ugly, text, blur, logo, static, low quality, "
    "noise, mutant, horror, film grain"
)
MIN_GPU_SECONDS = int(os.environ.get("MIN_GPU_SECONDS", "45"))
MAX_GPU_SECONDS = int(os.environ.get("MAX_GPU_SECONDS", "600"))
DEFAULT_ENHANCE_BUDGET = 80

SULPHUR_REPO = "SulphurAI/Sulphur-2-base"
SULPHUR_MODEL_FILE = "prompt_enhancer/sulphur_prompt_enhancer_model-q8_0.gguf"
SULPHUR_MMPROJ_FILE = "prompt_enhancer/mmproj-BF16.gguf"
SULPHUR_MODEL_DIR = ROOT / "sulphur_enhancer"
SULPHUR_MODEL_PATH = SULPHUR_MODEL_DIR / "sulphur_prompt_enhancer_model-q8_0.gguf"
SULPHUR_MMPROJ_PATH = SULPHUR_MODEL_DIR / "mmproj-BF16.gguf"

LLAMA_CPP_DIR = ROOT / "llama.cpp"
LLAMA_SERVER_BIN = LLAMA_CPP_DIR / "build" / "bin" / "llama-server"

# Canonical cache repo for the prebuilt llama-server binary. Pull is public and
# works for everyone (including duplicated spaces). Push only succeeds for the
# owner of this repo, so duplicated spaces never pollute it.
CACHE_REPO = "signsur4739379373/ltx-dependencies"
CACHE_BINARY_FILENAME = "llama-server"
CACHE_LIBS_TARBALL = "llama-server-libs.tar.gz"
CACHED_BINARY_PATH = ROOT / "llama-server-cached"
# CUDA shared libs the binary needs at runtime (the build box has CUDA 13 but
# the gpu runtime container may not expose it). We bundle them next to the
# binary and cache them so every boot has a matching runtime.
CACHED_LIBS_DIR = ROOT / "llama-server-libs"

_workflow_cache: dict[bool, dict[str, Any]] = {}
_comfy_ready = False
_nodes_ready = False
_enhancer_ready = False
_enhancer_lock = threading.Lock()
_enhancer_server_proc = None
ENHANCER_PORT = 18642


def _server_binary_path() -> pathlib.Path:
    """Return whichever llama-server binary is available (cached or built)."""
    if CACHED_BINARY_PATH.exists():
        return CACHED_BINARY_PATH
    return LLAMA_SERVER_BIN


def _have_server_artifacts() -> bool:
    """True if a usable binary + bundled libs already exist."""
    if not CACHED_LIBS_DIR.exists() or not any(CACHED_LIBS_DIR.glob("*.so*")):
        return False
    return CACHED_BINARY_PATH.exists() or LLAMA_SERVER_BIN.exists()


def _pull_cached_binary() -> bool:
    """Download prebuilt binary + bundled libs from the cache repo. Public, no token."""
    if CACHED_BINARY_PATH.exists() and CACHED_LIBS_DIR.exists():
        return True
    try:
        binary = pathlib.Path(hf_hub_download(repo_id=CACHE_REPO, filename=CACHE_BINARY_FILENAME))
        libs_tar = pathlib.Path(hf_hub_download(repo_id=CACHE_REPO, filename=CACHE_LIBS_TARBALL))
        shutil.copy2(binary, CACHED_BINARY_PATH)
        os.chmod(CACHED_BINARY_PATH, 0o755)
        CACHED_LIBS_DIR.mkdir(parents=True, exist_ok=True)
        import tarfile

        with tarfile.open(libs_tar, "r:gz") as tf:
            tf.extractall(CACHED_LIBS_DIR)
        print("[enhancer] pulled prebuilt llama-server + libs from cache repo", flush=True)
        return True
    except Exception as e:
        print(f"[enhancer] cache pull failed ({type(e).__name__}: {e}); will build", flush=True)
        return False


def _push_cached_binary() -> None:
    """Upload built binary + bundled libs tarball. Silently no-ops without write access."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("[enhancer] no token; skipping cache push", flush=True)
        return
    try:
        from huggingface_hub import HfApi

        # tar up the bundled libs
        libs_tar = ROOT / CACHE_LIBS_TARBALL
        import tarfile

        with tarfile.open(libs_tar, "w:gz") as tf:
            for so in CACHED_LIBS_DIR.glob("*"):
                tf.add(so, arcname=so.name)

        api = HfApi(token=token)
        api.create_repo(repo_id=CACHE_REPO, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(LLAMA_SERVER_BIN),
            path_in_repo=CACHE_BINARY_FILENAME,
            repo_id=CACHE_REPO,
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj=str(libs_tar),
            path_in_repo=CACHE_LIBS_TARBALL,
            repo_id=CACHE_REPO,
            repo_type="model",
        )
        print("[enhancer] pushed built llama-server + libs to cache repo", flush=True)
    except Exception as e:
        print(f"[enhancer] cache push failed ({type(e).__name__}: {e}); continuing", flush=True)


def _find_cuda13_lib_dir() -> pathlib.Path | None:
    """Locate the system CUDA 13 toolkit lib dir on the build box so the link
    step and runtime can resolve libcudart.so.13 (the box's nvcc is CUDA 13)."""
    candidates = [
        "/cuda-image/usr/local/cuda-13.0/targets/x86_64-linux/lib",
        "/cuda-image/usr/local/cuda-13.0/lib64",
        "/usr/local/cuda-13.0/targets/x86_64-linux/lib",
        "/usr/local/cuda-13.0/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda/lib64",
    ]
    for c in candidates:
        p = pathlib.Path(c)
        if (p / "libcudart.so").exists() or list(p.glob("libcudart.so.13*")):
            return p
    # last resort: search
    for base in ("/cuda-image/usr/local", "/usr/local"):
        bp = pathlib.Path(base)
        if not bp.exists():
            continue
        for found in bp.rglob("libcudart.so.13*"):
            return found.parent
    return None


def _build_llama_cpp() -> None:
    print("[enhancer] building llama.cpp from source...", flush=True)
    if not LLAMA_CPP_DIR.exists():
        _run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", str(LLAMA_CPP_DIR)])

    cuda_lib = _find_cuda13_lib_dir()
    if cuda_lib is None:
        raise RuntimeError("could not locate CUDA 13 libcudart on build box")
    print(f"[enhancer] using CUDA libs at {cuda_lib}", flush=True)

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{cuda_lib}:{env.get('LD_LIBRARY_PATH','')}"
    env["LIBRARY_PATH"] = f"{cuda_lib}:{env.get('LIBRARY_PATH','')}"

    def _run_env(cmd: list[str]) -> None:
        print("[setup]", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(LLAMA_CPP_DIR), check=True, env=env)

    shutil.rmtree(LLAMA_CPP_DIR / "build", ignore_errors=True)
    _run_env([
        "cmake", "-B", "build",
        "-DGGML_CUDA=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DLLAMA_CURL=OFF",
        "-DCMAKE_CUDA_ARCHITECTURES=86",
        # Explicitly point the linker at the CUDA 13 runtime libs so the final
        # link of llama-server resolves the cudart symbols.
        f"-DCMAKE_EXE_LINKER_FLAGS=-L{cuda_lib} -lcudart -Wl,-rpath,{cuda_lib}",
        f"-DCMAKE_SHARED_LINKER_FLAGS=-L{cuda_lib} -lcudart -Wl,-rpath,{cuda_lib}",
    ])
    build_cmd = ["cmake", "--build", "build", "--config", "Release", "--target", "llama-server"]
    try:
        _run_env(build_cmd + ["-j2"])
    except subprocess.CalledProcessError:
        print("[enhancer] -j2 build failed, retrying with -j1", flush=True)
        _run_env(build_cmd + ["-j1"])
    if not LLAMA_SERVER_BIN.exists():
        raise RuntimeError("llama-server binary not found after build")

    # Bundle the cuda runtime libs + llama.cpp's own .so outputs next to the
    # binary so it runs even when the build-time cuda path is gone at runtime.
    CACHED_LIBS_DIR.mkdir(parents=True, exist_ok=True)
    built_lib_dir = LLAMA_CPP_DIR / "build" / "bin"
    for so in built_lib_dir.glob("*.so*"):
        shutil.copy2(so, CACHED_LIBS_DIR / so.name)
    for pattern in ("libcudart.so*", "libcublas.so*", "libcublasLt.so*"):
        for so in cuda_lib.glob(pattern):
            target = CACHED_LIBS_DIR / so.name
            if not target.exists():
                shutil.copy2(so, target)
    print("[enhancer] llama.cpp built", flush=True)


def _ensure_llama_server() -> None:
    """Pull prebuilt binary + libs; if absent, build then push to seed the cache."""
    if _have_server_artifacts():
        return
    if _pull_cached_binary():
        return
    _build_llama_cpp()
    _push_cached_binary()


def _ensure_enhancer() -> None:
    """Prepare binary + sulphur enhancer weights. Sets _enhancer_ready; never raises."""
    global _enhancer_ready
    if _enhancer_ready:
        return
    # Kill switch for disk-constrained runtimes (e.g. free Colab, ~78 GB total):
    # skips the llama-server binary/build AND the sulphur weights (~13 GB).
    if os.environ.get("DISABLE_ENHANCER", "").strip().lower() in ("1", "true", "yes"):
        print("[enhancer] disabled via DISABLE_ENHANCER env var", flush=True)
        return
    try:
        _ensure_llama_server()
        SULPHUR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        for file_path, dest in [
            (SULPHUR_MODEL_FILE, SULPHUR_MODEL_PATH),
            (SULPHUR_MMPROJ_FILE, SULPHUR_MMPROJ_PATH),
        ]:
            if dest.exists():
                continue
            print(f"[enhancer] downloading {file_path}...", flush=True)
            downloaded = pathlib.Path(
                hf_hub_download(
                    repo_id=SULPHUR_REPO,
                    filename=file_path,
                    local_dir=str(SULPHUR_MODEL_DIR),
                    token=token,
                )
            )
            if downloaded.resolve() != dest.resolve():
                shutil.move(str(downloaded), str(dest))
        _enhancer_ready = True
        print("[enhancer] ready", flush=True)
    except Exception as e:
        print(f"[enhancer] setup failed, enhancer disabled ({type(e).__name__}: {e})", flush=True)
        _enhancer_ready = False


def _start_enhancer_server() -> None:
    global _enhancer_server_proc
    if _enhancer_server_proc is not None:
        try:
            _enhancer_server_proc.poll()
            if _enhancer_server_proc.returncode is None:
                return
        except Exception:
            pass
    server_bin = _server_binary_path()
    # Binary links against bundled CUDA + llama.cpp .so files; expose them.
    server_env = dict(os.environ)
    if CACHED_LIBS_DIR.exists():
        server_env["LD_LIBRARY_PATH"] = f"{CACHED_LIBS_DIR}:{server_env.get('LD_LIBRARY_PATH','')}"
    print(f"[enhancer] starting llama-server on port {ENHANCER_PORT}...", flush=True)
    _enhancer_server_proc = subprocess.Popen(
        [
            str(server_bin),
            "-m", str(SULPHUR_MODEL_PATH),
            "--mmproj", str(SULPHUR_MMPROJ_PATH),
            "-ngl", "99",
            "-c", "8192",
            "--flash-attn", "on",
            "--host", "127.0.0.1",
            "--port", str(ENHANCER_PORT),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=server_env,
    )
    for _ in range(60):
        time.sleep(1)
        try:
            r = http_requests.get(f"http://127.0.0.1:{ENHANCER_PORT}/health", timeout=2)
            if r.json().get("status") == "ok":
                print("[enhancer] server ready", flush=True)
                return
        except Exception:
            pass
    raise RuntimeError("enhancer server failed to start within 60s")


def _stop_enhancer_server() -> None:
    global _enhancer_server_proc
    if _enhancer_server_proc is not None:
        try:
            _enhancer_server_proc.terminate()
            _enhancer_server_proc.wait(timeout=10)
        except Exception:
            try:
                _enhancer_server_proc.kill()
            except Exception:
                pass
        _enhancer_server_proc = None


def _enhance_prompt_impl(image_paths: list[str], concept: str) -> str:
    """Call the sulphur llama-server enhancer with no system prompt so the
    model's trained behavior is preserved. Sends all provided images in a
    single chat message; the model decides how to attend to each."""
    with _enhancer_lock:
        _start_enhancer_server()

    content: list[dict[str, Any]] = []
    for path in image_paths:
        if not path:
            continue
        img = Image.open(path).convert("RGB")
        buf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(buf.name, format="JPEG", quality=85)
        with open(buf.name, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        os.unlink(buf.name)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": concept})

    payload = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2048,
        "temperature": 0.7,
    }
    resp = http_requests.post(
        f"http://127.0.0.1:{ENHANCER_PORT}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    data = resp.json()
    if "choices" not in data:
        raise RuntimeError(f"enhancer returned unexpected payload: {data}")
    text = data["choices"][0]["message"].get("content", "")
    if not text:
        text = data["choices"][0]["message"].get("reasoning_content", "")
    text = text.strip()
    img_count = sum(1 for c in content if c.get("type") == "image_url")
    print(f"[enhancer] enhanced prompt ({len(text)} chars, {img_count} images): {text}", flush=True)
    return text


def get_enhance_duration(
    image_path: str,
    prompt: str,
    enhance_budget: float = DEFAULT_ENHANCE_BUDGET,
    msr_ref2_path: str | None = None,
    msr_ref3_path: str | None = None,
    msr_ref4_path: str | None = None,
    msr_bg_path: str | None = None,
    progress: gr.Progress | None = None,
) -> int:
    return max(20, min(MAX_GPU_SECONDS, int(enhance_budget or DEFAULT_ENHANCE_BUDGET)))


@spaces.GPU(duration=get_enhance_duration)
def enhance_prompt(
    image_path: str,
    prompt: str,
    enhance_budget: float = DEFAULT_ENHANCE_BUDGET,
    msr_ref2_path: str | None = None,
    msr_ref3_path: str | None = None,
    msr_ref4_path: str | None = None,
    msr_bg_path: str | None = None,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> str:
    if not _enhancer_ready:
        raise gr.Error("prompt enhancer is not available on this instance")
    if not image_path:
        raise gr.Error("upload an image first")
    if not prompt.strip():
        raise gr.Error("write a concept/prompt first")
    image_paths = [image_path]
    for p in (msr_ref2_path, msr_ref3_path, msr_ref4_path, msr_bg_path):
        if p:
            image_paths.append(p)
    try:
        enhanced = _enhance_prompt_impl(image_paths, prompt.strip())
        if not enhanced:
            return prompt
        return enhanced
    except Exception:
        tb = traceback.format_exc()
        print(f"[enhancer] failed: {tb}", flush=True)
        raise gr.Error(f"enhancer failed: {tb[-500:]}")


def _ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(cmd: list[str], cwd: pathlib.Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("[setup]", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)


def _pip_install(args: list[str], check: bool = True) -> None:
    _run([sys.executable, "-m", "pip", "install", "--no-cache-dir", *args], check=check)


def _install_filtered_requirements(req_path: pathlib.Path) -> None:
    if not req_path.exists():
        return
    blocked = {"torch", "torchvision", "torchaudio", "transformers", "huggingface-hub", "accelerate"}
    safe: list[str] = []
    for line in req_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        low = item.lower().replace("_", "-")
        package = re.split(r"[<>=!~;\[\s]", low, maxsplit=1)[0]
        if package in blocked:
            continue
        safe.append(item)
    if safe:
        _pip_install(safe, check=False)


def _apply_comfy_utils_namespace_fix() -> None:
    utils_path = COMFY / "utils"
    utilities_path = COMFY / "utilities"
    if utils_path.exists() and not utilities_path.exists():
        utils_path.rename(utilities_path)

    replacements = [
        (re.compile(r"(^|\n)(\s*)from utils(\s|\.)"), r"\1\2from utilities\3"),
        (re.compile(r"(^|\n)(\s*)import utils(\s|\.|$)"), r"\1\2import utilities\3"),
    ]
    for path in COMFY.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = text
        for pattern, repl in replacements:
            updated = pattern.sub(repl, updated)
        updated = updated.replace("from utils import", "from utilities import")
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def _ensure_repo(path: pathlib.Path, url: str, commit: str | None = None) -> None:
    if not path.exists():
        _run(["git", "clone", "--depth", "1", url, str(path)])
    if commit:
        _run(["git", "fetch", "--depth", "1", "origin", commit], cwd=path, check=False)
        _run(["git", "checkout", commit], cwd=path, check=False)


def _ensure_comfy() -> None:
    global _comfy_ready
    if _comfy_ready:
        return

    _ensure_repo(
        COMFY,
        "https://github.com/comfyanonymous/ComfyUI.git",
        commit="4e1f7cb1db1c26bb9ee61cf1875776517e2abae8",
    )
    _install_filtered_requirements(COMFY / "requirements.txt")

    custom_root = COMFY / "custom_nodes"
    custom_root.mkdir(parents=True, exist_ok=True)
    for name, url in CUSTOM_NODES:
        node_path = custom_root / name
        _ensure_repo(node_path, url)
        _install_filtered_requirements(node_path / "requirements.txt")

    _install_kv_wrapper(COMFY)
    _apply_comfy_utils_namespace_fix()

    for folder in (
        "checkpoints",
        "text_encoders",
        "loras/ltx23",
        "upscale_models",
        "latent_upscale_models",
        "vae",
        "diffusion_models",
    ):
        (MODELS / folder).mkdir(parents=True, exist_ok=True)
    INPUT.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    _comfy_ready = True


def _link_or_copy(src: pathlib.Path, dest: pathlib.Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    if dest.is_symlink():
        dest.unlink()
    try:
        os.link(src, dest)
        return
    except OSError:
        pass
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _download_to_dest(repo: str, file_path: str, dest: pathlib.Path, token: str | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not dest.is_symlink():
        return
    if dest.is_symlink():
        dest.unlink()

    filename = pathlib.Path(file_path).name
    subfolder = str(pathlib.Path(file_path).parent)
    downloaded = pathlib.Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            subfolder=None if subfolder == "." else subfolder,
            local_dir=str(dest.parent),
            token=token,
        )
    )

    if downloaded.resolve() == dest.resolve():
        return
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(downloaded, dest)
    except OSError:
        _link_or_copy(downloaded, dest)


def _ensure_models(progress: gr.Progress | None = None) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    for index, item in enumerate(DOWNLOADS):
        if item.get("slot") in _SKIP_SLOT_LORAS:
            print(f"[models] skip {item['label']} (desmarcada en el notebook)", flush=True)
            continue
        dest = pathlib.Path(item["dest"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            continue
        if progress:
            progress(index / len(DOWNLOADS), desc=f"downloading {item['label']}")
        _download_to_dest(item["repo"], item["file"], dest, token)


def _init_comfy_nodes() -> None:
    global _nodes_ready
    if _nodes_ready:
        return

    comfy_path = str(COMFY)
    sys.path = [p for p in sys.path if p != comfy_path]
    sys.path.insert(0, comfy_path)
    for module_name in list(sys.modules):
        if module_name == "utils" or module_name.startswith("utils."):
            del sys.modules[module_name]
    os.chdir(COMFY)

    import execution
    import nodes
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    loop.run_until_complete(nodes.init_extra_nodes())
    _nodes_ready = True


def _node_widget_params(class_type: str) -> list[str]:
    import nodes

    cls = nodes.NODE_CLASS_MAPPINGS[class_type]
    params: list[str] = []
    inputs = cls.INPUT_TYPES()
    for group in ("required", "optional"):
        for name, spec in inputs.get(group, {}).items():
            typ = spec[0] if isinstance(spec, (tuple, list)) and spec else spec
            if isinstance(typ, (list, tuple)) or str(typ).upper() in {"FLOAT", "INT", "STRING", "BOOLEAN", "COMBO"}:
                params.append(name)
    return params


def _visual_widget_params(node: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for inp in node.get("inputs") or []:
        widget = inp.get("widget")
        if isinstance(widget, dict) and widget.get("name"):
            names.append(widget["name"])
    return names


def _convert_workflow(visual_path: str) -> dict[str, Any]:
    import nodes

    visual = json.loads(pathlib.Path(visual_path).read_text(encoding="utf-8"))
    visual_nodes = {int(node["id"]): node for node in visual.get("nodes", [])}

    primitive_values: dict[int, Any] = {}
    for node_id, node in visual_nodes.items():
        widgets = node.get("widgets_values") or []
        if node.get("type") == "JWStringToFloat" and widgets:
            try:
                primitive_values[node_id] = float(widgets[0])
            except (TypeError, ValueError):
                primitive_values[node_id] = widgets[0]
        elif node.get("type") == "easy loraNames" and widgets:
            primitive_values[node_id] = widgets[0]

    link_map: dict[int, Any] = {}
    for link in visual.get("links", []):
        link_id, src_node, src_slot, *_ = link
        link_map[int(link_id)] = primitive_values.get(int(src_node), [str(src_node), src_slot])

    set_sources: dict[str, Any] = {}
    set_node_sources: dict[int, Any] = {}
    for node in visual.get("nodes", []):
        if node.get("type") not in {"SetNode", "SetNodeAny"}:
            continue
        name = (node.get("widgets_values") or [""])[0]
        for inp in node.get("inputs") or []:
            link_id = inp.get("link")
            if link_id in link_map:
                set_sources[name] = link_map[link_id]
                set_node_sources[int(node["id"])] = link_map[link_id]

    changed = True
    while changed:
        changed = False
        for link_id, source in list(link_map.items()):
            if isinstance(source, list) and int(source[0]) in set_node_sources:
                replacement = set_node_sources[int(source[0])]
                if link_map[link_id] != replacement:
                    link_map[link_id] = replacement
                    changed = True
        for node in visual.get("nodes", []):
            if node.get("type") not in {"GetNode", "GetNodeAny"}:
                continue
            name = (node.get("widgets_values") or [""])[0]
            if name not in set_sources:
                continue
            for link_id, source in list(link_map.items()):
                if isinstance(source, list) and source[0] == str(node["id"]):
                    replacement = set_sources[name]
                    if link_map[link_id] != replacement:
                        link_map[link_id] = replacement
                        changed = True

    skip_ids = {
        617, 535, 548, 556, 591, 596, 550, 593, 274, 789, 780,
        551, 598, 549, 552, 755, 769,
    }
    rewires = {
        "510": {"latent_image": ["772", 0]},
        "744": {"samples": ["510", 1]},
        "802": {"latent_image": ["770", 0]},
        "740": {"samples": ["802", 1]},
        "597": {"images": ["740", 0]},
    }

    # Native AV: keep the concat/separate/decoder chain so 597.audio resolves
    # and the sampler operates on AV latents end-to-end. Audio is always on
    # (joint sampling adds no meaningful compute; toggling has no benefit).
    skip_ids = skip_ids - AUDIO_CHAIN_NODE_IDS
    # Drop the silent-only sampler/decoder rewires so the original AV path lives.
    rewires = {
        key: value for key, value in rewires.items()
        if key not in AUDIO_ONLY_REWIRE_KEYS
    }
    # Bypass node 789 (TwoWaySwitch) by hardwiring its selected input.
    rewires.update(AUDIO_BYPASS_REWIRES)
    skip_types = {
        "Note",
        "NoteNode",
        "MarkdownNote",
        "GetNode",
        "GetNodeAny",
        "SetNode",
        "SetNodeAny",
        "JWStringToFloat",
        "easy loraNames",
    }
    api: dict[str, Any] = {}

    for node in visual.get("nodes", []):
        node_id = int(node["id"])
        node_key = str(node_id)
        class_type = node["type"]
        if node_id in skip_ids or class_type in skip_types:
            continue
        if class_type not in nodes.NODE_CLASS_MAPPINGS:
            print(f"[workflow] skipping missing node type {class_type} ({node_key})", flush=True)
            continue

        inputs: dict[str, Any] = dict(rewires.get(node_key, {}))
        for inp in node.get("inputs") or []:
            link_id = inp.get("link")
            if link_id is None or link_id not in link_map:
                continue
            source = link_map[link_id]
            if isinstance(source, list) and int(source[0]) in skip_ids:
                continue
            inputs.setdefault(inp["name"], source)

        widgets = node.get("widgets_values") or []
        if class_type == "Power Lora Loader (rgthree)":
            # We rewrite rgthree's Power Lora Loader to phazei's MultiLoRALoader
            # in LTX mode so each lora has separate video/audio strength control
            # (Vid, V2A, Aud, A2V, Other per-tensor-pattern multipliers on top of
            # the global STR). Same output signature (model, clip), so downstream
            # connections work unchanged. Lora list lives in the lora_data JSON
            # string; _inject_optional_loras populates it later. OmniNFT entries
            # from the template are dropped here (exposed separately via the
            # OPTIONAL_LORAS sliders).
            class_type = "MultiLoRALoader"
            inputs["lora_data"] = "[]"
            inputs["ltx_mode"] = True
        elif isinstance(widgets, dict):
            for key, value in widgets.items():
                if key != "videopreview":
                    inputs.setdefault(key, value)
        elif widgets:
            param_names = _visual_widget_params(node) or _node_widget_params(class_type)
            for key, value in zip(param_names, widgets):
                inputs.setdefault(key, value)

        if class_type == "LTX2LoraLoaderAdvanced":
            widget_values = node.get("widgets_values") or []
            if widget_values:
                inputs["lora_name"] = widget_values[0]
                inputs["opt_lora_path"] = str(MODELS / "loras" / widget_values[0].replace("\\", "/"))
            else:
                inputs.setdefault("opt_lora_path", "")
            inputs.setdefault("blocks", "")
            if inputs.get("lora_name") is None:
                inputs["lora_name"] = ""

        api[node_key] = {"class_type": class_type, "inputs": inputs}

    return api


def _workflow_template() -> dict[str, Any]:
    if "default" not in _workflow_cache:
        path = hf_hub_download(
            repo_id=WORKFLOW_REPO,
            repo_type="model",
            filename=WORKFLOW_FILENAME,
            revision=WORKFLOW_REVISION,
        )
        _workflow_cache["default"] = _convert_workflow(path)
    return json.loads(json.dumps(_workflow_cache["default"]))


def _convert_runexx_workflow(visual_path: str) -> dict[str, Any]:
    """Convert the bundled runexx visual workflow to API form, patching the
    split UNET/VAE/CLIP loader chain to use our 10Eros checkpoint and stripping
    the GGUF parallel path + unused preview/distilled nodes.

    Pre-conversion patches:
      59  UNETLoader        -> CheckpointLoaderSimple (10Eros)
      57  DualCLIPLoader    -> LTXAVTextEncoderLoader (gemma + 10Eros)
      53  VAELoaderKJ       -> LTXVAudioVAELoader (10Eros)
    Link rewires:
      56  VAELoader (video)    -> outputs replaced with CheckpointLoaderSimple slot 2
      1245 UUID conditioning   -> outputs replaced with IC-LoRA guide pass 1 slots 0/1
      1222 UUID image size     -> outputs replaced with INTConstant width/height (166/167)
      1235 ComfySwitchNode     -> outputs replaced with sampler pass 2 (139) direct
    Skipped nodes:
      55 (VAELoader preview), 60 (LoraLoaderModelOnly distilled),
      1257/1256 (GGUF parallel path), 56, 1222, 1245, 1235 (replaced via rewires).
    """
    import nodes

    visual = json.loads(pathlib.Path(visual_path).read_text(encoding="utf-8"))

    for node in visual.get("nodes", []):
        nid = int(node["id"])
        if nid == RUNEXX_NODE_UNET_LOADER:
            node["type"] = "CheckpointLoaderSimple"
            node["widgets_values"] = ["10Eros_v1-fp8mixed_learned.safetensors"]
        elif nid == RUNEXX_NODE_CLIP_LOADER:
            node["type"] = "LTXAVTextEncoderLoader"
            node["widgets_values"] = [
                "gemma_3_12B_it_fp8_scaled.safetensors",
                "10Eros_v1-fp8mixed_learned.safetensors",
                "default",
            ]
        elif nid == RUNEXX_NODE_VAE_AUDIO:
            node["type"] = "LTXVAudioVAELoader"
            node["widgets_values"] = ["10Eros_v1-fp8mixed_learned.safetensors"]

    # Skip the dead loader/preview/parallel nodes AND the UUID stand-ins which
    # we replace via the link rewire pass below.
    skip_ids = {
        RUNEXX_NODE_VAE_VIDEO,
        RUNEXX_NODE_VAE_TINY,
        RUNEXX_NODE_DISTILLED_LORA,
        RUNEXX_NODE_GGUF_UNET,
        RUNEXX_NODE_GGUF_CLIP,
        RUNEXX_NODE_UUID_IMAGESIZE,
        RUNEXX_NODE_UUID_CONDITIONING,
        RUNEXX_NODE_SAMPLER_SWITCH,
    }
    skip_types = {
        "Note", "NoteNode", "MarkdownNote",
        "GetNode", "GetNodeAny", "SetNode", "SetNodeAny",
        "JWStringToFloat", "easy loraNames",
        # PathchSageAttentionKJ requires sage-attention / triton; the workflow
        # works without it (slower attention) so we skip rather than fail.
        "PathchSageAttentionKJ",
    }

    visual_nodes = {int(n["id"]): n for n in visual.get("nodes", [])}
    primitive_values: dict[int, Any] = {}
    for nid, n in visual_nodes.items():
        widgets = n.get("widgets_values") or []
        if n.get("type") == "JWStringToFloat" and widgets:
            try:
                primitive_values[nid] = float(widgets[0])
            except (TypeError, ValueError):
                primitive_values[nid] = widgets[0]
        elif n.get("type") == "easy loraNames" and widgets:
            primitive_values[nid] = widgets[0]

    # Rewires applied at link-resolution time. Map keyed by (src_node_id,
    # src_slot) -> new [src_node_id, src_slot]. These replace dead UUID nodes
    # and the deleted video VAE loader with live equivalents.
    link_rewires: dict[tuple[int, int], list] = {
        # Deleted VAELoader (video). Consumers fed [56, 0]; rewire to
        # CheckpointLoaderSimple's VAE output (slot 2).
        (RUNEXX_NODE_VAE_VIDEO, 0): [str(RUNEXX_NODE_UNET_LOADER), 2],
        # UUID image-size (1222) had 4 INT outputs: 0=height_first,
        # 1=width_first, 2=width_final, 3=height_final. Map width/height to
        # INTConstant 166/167.
        (RUNEXX_NODE_UUID_IMAGESIZE, 0): [str(RUNEXX_NODE_HEIGHT_CONST), 0],
        (RUNEXX_NODE_UUID_IMAGESIZE, 1): [str(RUNEXX_NODE_WIDTH_CONST), 0],
        (RUNEXX_NODE_UUID_IMAGESIZE, 2): [str(RUNEXX_NODE_WIDTH_CONST), 0],
        (RUNEXX_NODE_UUID_IMAGESIZE, 3): [str(RUNEXX_NODE_HEIGHT_CONST), 0],
        # UUID conditioning (1245) feeds pass-1 CropGuides positive/negative.
        # Canonical pattern: those come from the pass-1 IC-LoRA guide.
        (RUNEXX_NODE_UUID_CONDITIONING, 0): [str(RUNEXX_NODE_ICLORA_GUIDE_P1), 0],
        (RUNEXX_NODE_UUID_CONDITIONING, 1): [str(RUNEXX_NODE_ICLORA_GUIDE_P1), 1],
        # Sampler switch (1235) gated between pass-1 and pass-2 sampler
        # outputs; we hardcode the pass-2 path (which produces upscaled output).
        (RUNEXX_NODE_SAMPLER_SWITCH, 0): [str(RUNEXX_NODE_SAMPLER_P2), 0],
    }

    def _apply_rewire(source):
        if not (isinstance(source, list) and len(source) >= 2):
            return source
        try:
            key = (int(source[0]), int(source[1]))
        except (TypeError, ValueError):
            return source
        return link_rewires.get(key, source)

    link_map: dict[int, Any] = {}
    for link in visual.get("links", []):
        if not (isinstance(link, list) and len(link) >= 3):
            continue
        link_id, src_node, src_slot = link[0], link[1], link[2]
        if int(src_node) in primitive_values:
            link_map[int(link_id)] = primitive_values[int(src_node)]
            continue
        source = [str(src_node), src_slot]
        source = _apply_rewire(source)
        link_map[int(link_id)] = source

    # Resolve SetNode -> GetNode chains.
    set_sources: dict[str, Any] = {}
    set_node_sources: dict[int, Any] = {}
    for n in visual.get("nodes", []):
        if n.get("type") not in {"SetNode", "SetNodeAny"}:
            continue
        name = (n.get("widgets_values") or [""])[0]
        for inp in n.get("inputs") or []:
            link_id = inp.get("link")
            if link_id in link_map:
                set_sources[name] = link_map[link_id]
                set_node_sources[int(n["id"])] = link_map[link_id]

    changed = True
    while changed:
        changed = False
        for link_id, source in list(link_map.items()):
            if isinstance(source, list) and len(source) >= 2:
                try:
                    src_id = int(source[0])
                except (TypeError, ValueError):
                    continue
                if src_id in set_node_sources:
                    replacement = set_node_sources[src_id]
                    if link_map[link_id] != replacement:
                        link_map[link_id] = replacement
                        changed = True
        for n in visual.get("nodes", []):
            if n.get("type") not in {"GetNode", "GetNodeAny"}:
                continue
            name = (n.get("widgets_values") or [""])[0]
            if name not in set_sources:
                continue
            get_id = int(n["id"])
            for link_id, source in list(link_map.items()):
                if isinstance(source, list) and len(source) >= 2:
                    try:
                        if int(source[0]) == get_id:
                            replacement = set_sources[name]
                            if link_map[link_id] != replacement:
                                link_map[link_id] = replacement
                                changed = True
                    except (TypeError, ValueError):
                        continue

    api: dict[str, Any] = {}
    for n in visual.get("nodes", []):
        nid = int(n["id"])
        node_key = str(nid)
        class_type = n["type"]
        if nid in skip_ids or class_type in skip_types:
            continue
        if class_type not in nodes.NODE_CLASS_MAPPINGS:
            print(f"[runexx-workflow] skipping unknown node {class_type} ({node_key})", flush=True)
            continue

        inputs: dict[str, Any] = {}
        for inp in n.get("inputs") or []:
            link_id = inp.get("link")
            if link_id is None or link_id not in link_map:
                continue
            source = link_map[link_id]
            if isinstance(source, list) and len(source) >= 2:
                try:
                    if int(source[0]) in skip_ids:
                        continue
                except (TypeError, ValueError):
                    pass
            inputs.setdefault(inp["name"], source)

        widgets = n.get("widgets_values") or []
        if class_type == "Power Lora Loader (rgthree)":
            # Same rewrite as the primary converter: rgthree -> MultiLoRALoader
            # in LTX mode for per-modality strength control.
            class_type = "MultiLoRALoader"
            inputs["lora_data"] = "[]"
            inputs["ltx_mode"] = True
        elif isinstance(widgets, dict):
            for key, value in widgets.items():
                if key != "videopreview":
                    inputs.setdefault(key, value)
        elif widgets:
            param_names = _visual_widget_params(n) or _node_widget_params(class_type)
            for key, value in zip(param_names, widgets):
                inputs.setdefault(key, value)

        api[node_key] = {"class_type": class_type, "inputs": inputs}

    return api


def _runexx_workflow_template() -> dict[str, Any]:
    if "runexx" not in _workflow_cache:
        path = str(ROOT / RUNEXX_WORKFLOW_FILE)
        _workflow_cache["runexx"] = _convert_runexx_workflow(path)
    return json.loads(json.dumps(_workflow_cache["runexx"]))


def _inject_runexx_params(
    workflow: dict[str, Any],
    *,
    ref1_image_name: str,
    ref2_image_name: str | None,
    bg_image_name: str | None,
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    frames: int,
    msr_frame_count: int,
) -> dict[str, Any]:
    """Patch user inputs into the converted runexx workflow.

    Maps UI inputs to the bundled workflow's CLIPTextEncode / LoadImage /
    RandomNoise / INTConstant / LiconMSR / EmptyLTXVLatentVideo widgets.
    """
    def _set_input(node_id: int, key: str, value: Any) -> None:
        node = workflow.get(str(node_id))
        if node is None:
            return
        node["inputs"][key] = value

    # Prompt text encoders (positive / negative).
    _set_input(RUNEXX_NODE_CLIPTEXT_POS, "text", prompt)
    _set_input(RUNEXX_NODE_CLIPTEXT_NEG, "text", negative_prompt)

    # Reference + background image uploads.
    _set_input(RUNEXX_NODE_LOAD_IMAGE_REF1, "image", ref1_image_name)
    if ref2_image_name:
        _set_input(RUNEXX_NODE_LOAD_IMAGE_REF2, "image", ref2_image_name)
    else:
        # Fall back to ref1 when only one subject reference is provided so
        # the LiconMSR slot stays populated.
        _set_input(RUNEXX_NODE_LOAD_IMAGE_REF2, "image", ref1_image_name)
    if bg_image_name:
        _set_input(RUNEXX_NODE_LOAD_IMAGE_BG, "image", bg_image_name)
    else:
        _set_input(RUNEXX_NODE_LOAD_IMAGE_BG, "image", ref1_image_name)

    # Seed: RandomNoise widget names are noise_seed/control_after_generate.
    _set_input(RUNEXX_NODE_RANDOM_NOISE, "noise_seed", int(seed))

    # Dimensions via the INTConstant widgets feeding the SetNode chain.
    _set_input(RUNEXX_NODE_WIDTH_CONST, "value", int(width))
    _set_input(RUNEXX_NODE_HEIGHT_CONST, "value", int(height))

    # LiconMSR widgets carry width / height / frame_count.
    _set_input(RUNEXX_NODE_LICON_MSR, "width", int(width))
    _set_input(RUNEXX_NODE_LICON_MSR, "height", int(height))
    _set_input(RUNEXX_NODE_LICON_MSR, "frame_count", int(msr_frame_count))

    # EmptyLTXVLatentVideo: extend by msr_frame_count so the requested
    # duration survives after LTXVCropGuides strips conditioning frames.
    raw_total = max(9, int(frames) + int(msr_frame_count))
    n_block = (raw_total - 1 + 7) // 8
    extended_length = max(9, n_block * 8 + 1)
    _set_input(RUNEXX_NODE_EMPTY_LATENT, "width", int(width))
    _set_input(RUNEXX_NODE_EMPTY_LATENT, "height", int(height))
    _set_input(RUNEXX_NODE_EMPTY_LATENT, "length", int(extended_length))

    return workflow


def _set_slider(workflow: dict[str, Any], node_id: str, value: int | float) -> None:
    if node_id not in workflow:
        return
    for key, old in list(workflow[node_id]["inputs"].items()):
        if not isinstance(old, list):
            workflow[node_id]["inputs"][key] = value


def _inject_params(
    workflow: dict[str, Any],
    *,
    preset: str,
    image_name: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    frames: int,
    mode: str,
    face_bbox: str,
    likeness_strength: float,
    likeness_anchor_strength: float,
    latent_anchor_strength: float,
    first_frame_strength: float,
    sulphur_lora_strength: float = 0.15,
    sulphur_v1_lora_strength: float = 0.15,
    vbvr_lora_strength: float = 0.5,
    dreamly_lora_strength: float = 0.6,
    synth_lora_strength: float = 0.0,
    plora_lora_strength: float = 0.0,
    singularity_lora_strength: float = 0.3,
    omninft_lora_strength: float = 0.8,
    omninft_bf16_lora_strength: float = 0.0,
    better_motion_lora_strength: float = 0.0,
    physics_v2_lora_strength: float = 0.0,
    hardcut_lora_strength: float = 0.0,
    transition_lora_strength: float = 0.15,
    slot7_lora_strength: float = 0.0,
    sulphur_audio_strength: float = 0.15,
    sulphur_v1_audio_strength: float = 0.15,
    vbvr_audio_strength: float = 0.5,
    dreamly_audio_strength: float = 0.6,
    synth_audio_strength: float = 0.0,
    plora_audio_strength: float = 0.0,
    singularity_audio_strength: float = 0.3,
    omninft_audio_strength: float = 0.8,
    omninft_bf16_audio_strength: float = 0.0,
    better_motion_audio_strength: float = 0.0,
    physics_v2_audio_strength: float = 0.0,
    hardcut_audio_strength: float = 0.0,
    transition_audio_strength: float = 0.0,
    slot7_audio_strength: float = 0.0,
    slot1_lora_file: str = HARDCUT_LORA_FILENAME,
    slot2_lora_file: str = SYNTH_LORA_FILENAME,
    slot3_lora_file: str = PLORA_LORA_FILENAME,
    slot4_lora_file: str = OMNINFT_BF16_LORA_FILENAME,
    slot5_lora_file: str = BETTER_MOTION_LORA_FILENAME,
    slot6_lora_file: str = PHYSICS_V2_LORA_FILENAME,
    slot7_lora_file: str = LORA_NONE,
    cache_at_step: int = 0,
    cache_warmup: int = 400,
    energy_threshold: float = 0.3,
    anchor_similarity_threshold: float = 0.3,
    sigma_string: str = _SIGMA_TUNED,
    msr_enabled: bool = False,
    msr_ref2_name: str | None = None,
    msr_ref3_name: str | None = None,
    msr_ref4_name: str | None = None,
    msr_bg_name: str | None = None,
    msr_frame_count: int = 41,
    msr_guide_strength: float = 1.0,
    msr_lora_strength: float = 0.7,
    prompt_relay_enabled: bool = False,
    prompt_segments: str = "",
    scene_chain_enabled: bool = False,
    scene_chain_prompt: str = "",
    scene_chain_max_scenes: int = 2,
    scene_chain_frame_overlap: int = 8,
    scene_chain_mid_guide: bool = True,
    scene_chain_mid_guide_strength: float = 0.25,
    kv_enabled: bool = False,
    kv_strength: float = 1.0,
    audio_ref_enabled: bool = False,
    audio_ref_filename: str | None = None,
    audio_ref_guidance_scale: float = 3.0,
    audio_ref_stem_sep: bool = False,
    audio_ref_normalize: bool = True,
) -> dict[str, Any]:
    # MSR (multi-reference) mode patches the workflow heavily - bypasses the
    # likeness/anchor system, inserts IC-LoRA conditioning, adds crop guides
    # to the decode path. Done BEFORE everything else so subsequent injections
    # see the patched workflow.
    if msr_enabled:
        _inject_msr(
            workflow,
            width=width,
            height=height,
            output_frames=int(frames),
            frame_count=int(msr_frame_count),
            guide_strength=float(msr_guide_strength),
            msr_lora_strength=float(msr_lora_strength),
            ref1_image_name=image_name,
            ref2_image_name=msr_ref2_name,
            ref3_image_name=msr_ref3_name,
            ref4_image_name=msr_ref4_name,
            bg_image_name=msr_bg_name,
        )
    # Prompt relay: timeline-based prompt routing. Disabled in MSR mode
    # because MSR already rewires the model + conditioning chain in
    # incompatible ways. Legacy second ranges are converted to the plugin's
    # smart prompt format; native smart syntax is passed through unchanged.
    scene_chain_scenes = _parse_scene_chain_scenes(
        scene_chain_prompt, max_scenes=int(scene_chain_max_scenes)
    ) if scene_chain_enabled and not msr_enabled else []
    if prompt_relay_enabled and not scene_chain_scenes and not msr_enabled and prompt_segments:
        smart_prompt = _prompt_relay_smart_prompt(prompt_segments, float(frames) / 24.0)
        if smart_prompt:
            _inject_prompt_relay(
                workflow,
                smart_prompt=smart_prompt,
                global_prompt=prompt,
            )
    # K/V identity conditioning. Disabled in MSR mode (model chain already
    # rewired). Stacks cleanly on top of prompt relay if both are active -
    # K/V reads whatever upstream model is currently wired into power loader,
    # which may be the relay node's output.
    if kv_enabled and not msr_enabled:
        _inject_kv_conditioning(workflow, strength=float(kv_strength))
    # Audio reference: voice ID transfer. Splices LTXVReferenceAudio between
    # PowerLora and downstream, also patching conditioning. Disabled in MSR
    # mode (heavily-rewired chain) and skipped if no audio uploaded.
    if (audio_ref_enabled and audio_ref_filename and not msr_enabled and not scene_chain_scenes):
        _inject_audio_reference(
            workflow,
            audio_filename=audio_ref_filename,
            guidance_scale=float(audio_ref_guidance_scale),
            stem_sep=bool(audio_ref_stem_sep),
            normalize_audio=bool(audio_ref_normalize),
        )
    # Refine-pass sigmas. original=workflow default. tuned=drops the 0.715
    # high-sigma step. custom=validated upstream string.
    _inject_refine_sigmas(workflow, _validate_sigmas(sigma_string) if sigma_string and sigma_string.strip() else _SIGMA_TUNED)
    # cache_at_step 0 = auto-align to frame count (round(frames/40), clamped
    # 2-12). The cache step controls when the latent anchor's conditioning
    # kicks in; misalignment with frame count weakens identity at longer
    # durations.
    if int(cache_at_step) <= 0:
        resolved_cache_step = max(2, min(12, round(frames / 40)))
    else:
        resolved_cache_step = int(cache_at_step)
    workflow[NODE_LOAD_IMAGE]["inputs"]["image"] = image_name
    _inject_optional_loras(
        workflow,
        video_strengths={
            "lora_sulphur": sulphur_lora_strength,
            "lora_sulphur_v1": sulphur_v1_lora_strength,
            "lora_vbvr": vbvr_lora_strength,
            "lora_dreamly": dreamly_lora_strength,
            "lora_synth": synth_lora_strength,
            "lora_plora": plora_lora_strength,
            "lora_singularity": singularity_lora_strength,
            "lora_omninft": omninft_lora_strength,
            "lora_omninft_bf16": omninft_bf16_lora_strength,
            "lora_better_motion": better_motion_lora_strength,
            "lora_physics_v2": physics_v2_lora_strength,
            "lora_hardcut": hardcut_lora_strength,
            "lora_transition": transition_lora_strength,
            "lora_slot7": slot7_lora_strength,
            "lora_custom1": custom1_lora_strength,
            "lora_custom2": custom2_lora_strength,
            "lora_custom3": custom3_lora_strength,
            "lora_custom4": custom4_lora_strength,
            "lora_custom5": custom5_lora_strength,
            "lora_custom6": custom6_lora_strength,
        },
        audio_strengths={
            "lora_sulphur": sulphur_audio_strength,
            "lora_sulphur_v1": sulphur_v1_audio_strength,
            "lora_vbvr": vbvr_audio_strength,
            "lora_dreamly": dreamly_audio_strength,
            "lora_synth": synth_audio_strength,
            "lora_plora": plora_audio_strength,
            "lora_singularity": singularity_audio_strength,
            "lora_omninft": omninft_audio_strength,
            "lora_omninft_bf16": omninft_bf16_audio_strength,
            "lora_better_motion": better_motion_audio_strength,
            "lora_physics_v2": physics_v2_audio_strength,
            "lora_hardcut": hardcut_audio_strength,
            "lora_transition": transition_audio_strength,
            "lora_slot7": slot7_audio_strength,
            "lora_custom1": custom1_audio_strength,
            "lora_custom2": custom2_audio_strength,
            "lora_custom3": custom3_audio_strength,
            "lora_custom4": custom4_audio_strength,
            "lora_custom5": custom5_audio_strength,
            "lora_custom6": custom6_audio_strength,
        },
        file_overrides={
            "lora_hardcut": slot1_lora_file,
            "lora_synth": slot2_lora_file,
            "lora_plora": slot3_lora_file,
            "lora_omninft_bf16": slot4_lora_file,
            "lora_better_motion": slot5_lora_file,
            "lora_physics_v2": slot6_lora_file,
            "lora_slot7": slot7_lora_file,
            "lora_custom1": custom1_lora_file,
            "lora_custom2": custom2_lora_file,
            "lora_custom3": custom3_lora_file,
            "lora_custom4": custom4_lora_file,
            "lora_custom5": custom5_lora_file,
            "lora_custom6": custom6_lora_file,
        },
    )
    workflow[NODE_POSITIVE]["inputs"]["text"] = prompt
    workflow[NODE_NEGATIVE]["inputs"]["text"] = negative_prompt
    workflow[NODE_SEED]["inputs"]["seed"] = seed
    _set_slider(workflow, NODE_WIDTH, width)
    _set_slider(workflow, NODE_HEIGHT, height)
    _set_slider(workflow, NODE_LENGTH, max(1, frames - 1))
    _set_slider(workflow, NODE_FIRST_FRAME, first_frame_strength)

    guide = workflow.get(NODE_LIKENESS_GUIDE, {}).get("inputs", {})
    anchor = workflow.get(NODE_LIKENESS_ANCHOR, {}).get("inputs", {})
    latent_anchor = workflow.get(NODE_LATENT_ANCHOR, {}).get("inputs", {})

    if mode == "anchor only":
        guide["strength"] = 0.0
        guide["face_detect"] = "none"
        guide["face_bbox_within_reference"] = ""
        anchor["strength"] = 0.0
        anchor["bypass"] = True
        anchor["frame_0_bbox"] = ""
        anchor["override_face_bbox"] = ""
        latent_anchor["strength"] = latent_anchor_strength
        latent_anchor["cache_at_step"] = resolved_cache_step
        latent_anchor["cache_warmup"] = int(cache_warmup)
        latent_anchor["energy_threshold"] = float(energy_threshold)
        latent_anchor["similarity_threshold"] = float(anchor_similarity_threshold)

    elif preset == "original":
        guide["strength"] = likeness_strength
        guide["placement_mode"] = "silent_reference"
        guide["face_detect"] = "manual"
        guide["reference_mask_mode"] = "bbox_only"
        guide["face_padding"] = 0.15
        guide["crf"] = 24
        guide["blur_radius"] = 0
        guide["interpolation"] = "area"
        guide["crop"] = "center"
        guide["attention_strength"] = 1
        guide["emit_latent"] = "passthrough"
        guide["debug"] = False

        anchor["strength"] = likeness_anchor_strength
        anchor["reference_source"] = "auto"
        anchor["similarity_threshold"] = float(anchor_similarity_threshold)
        anchor["decay_with_distance"] = 0
        anchor["bypass"] = False
        anchor["debug"] = False
        anchor["advanced_mode"] = False
        anchor["depth_curve"] = "middle"
        anchor["block_index_filter"] = ""
        anchor["similarity_sharpness"] = 8
        anchor["override_face_bbox"] = ""
        anchor["skip_when_sigma_above"] = 0
        anchor["pull_mode"] = "directional"
        anchor["late_block_falloff"] = 0.4

        latent_anchor["strength"] = latent_anchor_strength
        latent_anchor["cache_at_step"] = resolved_cache_step
        latent_anchor["similarity_threshold"] = float(anchor_similarity_threshold)
        latent_anchor["decay_with_distance"] = 0.15
        latent_anchor["energy_threshold"] = float(energy_threshold)
        latent_anchor["bypass"] = False
        latent_anchor["debug"] = False
        latent_anchor["advanced_mode"] = True
        latent_anchor["cache_mode"] = "schedule"
        latent_anchor["forwards_per_step"] = 2
        latent_anchor["cache_warmup"] = int(cache_warmup)
        latent_anchor["anchor_frame"] = 0
        latent_anchor["depth_curve"] = "flat"
        latent_anchor["block_index_filter"] = ""

        if mode == "manual bbox" and face_bbox.strip():
            guide["face_bbox_within_reference"] = face_bbox.strip()
            anchor["frame_0_bbox"] = face_bbox.strip()

    else:
        guide["strength"] = likeness_strength
        guide["placement_mode"] = "silent_reference"
        anchor["strength"] = likeness_anchor_strength
        latent_anchor["strength"] = latent_anchor_strength
        guide["face_detect"] = "manual" if mode == "manual bbox" else "auto"
        guide["face_bbox_within_reference"] = face_bbox.strip()
        guide["reference_mask_mode"] = "bbox_softfade"
        guide["face_padding"] = 0.15
        guide["crf"] = 24
        guide["blur_radius"] = 0
        guide["interpolation"] = "area"
        guide["crop"] = "center"
        guide["attention_strength"] = 1
        guide["emit_latent"] = "passthrough"
        guide["debug"] = False

        anchor["reference_source"] = "auto"
        anchor["similarity_threshold"] = float(anchor_similarity_threshold)
        anchor["decay_with_distance"] = 0
        anchor["bypass"] = False
        anchor["debug"] = False
        anchor["advanced_mode"] = True
        anchor["depth_curve"] = "flat"
        anchor["block_index_filter"] = ""
        anchor["similarity_sharpness"] = 6
        anchor["override_face_bbox"] = face_bbox.strip()
        anchor["skip_when_sigma_above"] = 0
        anchor["pull_mode"] = "directional"
        anchor["late_block_falloff"] = 0.4

        latent_anchor["cache_at_step"] = resolved_cache_step
        latent_anchor["similarity_threshold"] = float(anchor_similarity_threshold)
        latent_anchor["decay_with_distance"] = 0.15
        latent_anchor["energy_threshold"] = float(energy_threshold)
        latent_anchor["bypass"] = False
        latent_anchor["debug"] = False
        latent_anchor["advanced_mode"] = True
        latent_anchor["cache_mode"] = "schedule"
        latent_anchor["forwards_per_step"] = 2
        latent_anchor["cache_warmup"] = int(cache_warmup)
        latent_anchor["anchor_frame"] = 0
        latent_anchor["depth_curve"] = "flat"
        latent_anchor["block_index_filter"] = ""

    if scene_chain_scenes:
        _inject_scene_chain(
            workflow,
            scenes=scene_chain_scenes,
            global_prompt=prompt,
            total_frames=int(frames),
            frame_overlap=int(scene_chain_frame_overlap),
            mid_scene_guide=bool(scene_chain_mid_guide),
            mid_scene_guide_strength=float(scene_chain_mid_guide_strength),
        )

    return workflow


OPTIONAL_LORAS = {
    "lora_sulphur": SULPHUR_LORA_FILENAME,
    "lora_sulphur_v1": SULPHUR_V1_LORA_FILENAME,
    "lora_vbvr": VBVR_LORA_FILENAME,
    "lora_dreamly": DREAMLY_LORA_FILENAME,
    "lora_synth": SYNTH_LORA_FILENAME,
    "lora_plora": PLORA_LORA_FILENAME,
    "lora_singularity": SINGULARITY_LORA_FILENAME,
    "lora_omninft": OMNINFT_LORA_FILENAME,
    "lora_omninft_bf16": OMNINFT_BF16_LORA_FILENAME,
    "lora_better_motion": BETTER_MOTION_LORA_FILENAME,
    "lora_physics_v2": PHYSICS_V2_LORA_FILENAME,
    "lora_hardcut": HARDCUT_LORA_FILENAME,
    "lora_transition": TRANSITION_LORA_FILENAME,
    "lora_slot7": LORA_NONE,
    "lora_custom1": LORA_NONE,
    "lora_custom2": LORA_NONE,
    "lora_custom3": LORA_NONE,
    "lora_custom4": LORA_NONE,
    "lora_custom5": LORA_NONE,
    "lora_custom6": LORA_NONE,
}


def _inject_optional_loras(
    workflow: dict[str, Any],
    video_strengths: dict[str, float],
    audio_strengths: dict[str, float] | None = None,
    file_overrides: dict[str, str] | None = None,
) -> None:
    """Populate the MultiLoRALoader's lora_data JSON string.

    LTX-mode entry format (per phazei's dispatch): per-key alpha is multiplied
    by the modality factor matching the tensor name pattern, then the global
    `str` applies on top. vid covers main video attn/ff.net tensors, aud covers
    audio_attn / audio_ff.net, v2a / a2v cover cross-modal attn. Setting aud
    independent of vid lets a non-audio-trained lora influence video without
    distorting the audio stream. Disabled (skipped) when video_strength <= 0
    and audio_strength <= 0. Idempotent.
    """
    node = workflow.get(NODE_POWER_LORA)
    if node is None:
        return
    audio_strengths = audio_strengths or {}
    file_overrides = file_overrides or {}
    entries: list[dict[str, Any]] = []
    for key, filename in OPTIONAL_LORAS.items():
        # Per-slot dropdown override: swap the file at runtime, or LORA_NONE
        # to disable the slot regardless of slider values.
        filename = file_overrides.get(key) or filename
        if not filename or filename == LORA_NONE:
            continue
        vid = float(video_strengths.get(key, 0.0) or 0.0)
        aud = float(audio_strengths.get(key, vid) or 0.0)
        if vid <= 0 and aud <= 0:
            continue
        entries.append({
            "lora": filename,
            "on": True,
            "str": 1.0,
            "vid": vid,
            "v2a": vid,
            "aud": aud,
            "a2v": vid,
            "other": vid,
        })
    node["inputs"]["lora_data"] = json.dumps(entries)
    node["inputs"]["ltx_mode"] = True


def _validate_sigmas(s: str) -> str:
    """Parse and validate a comma-separated refine sigma string.

    Returns the cleaned canonical string on success. Raises ValueError with a
    user-readable message on any problem so the caller can surface it via
    gr.Error before any GPU time is consumed.
    """
    if not s or not s.strip():
        raise ValueError("custom sigmas: empty input")
    parts = [x.strip() for x in s.replace(";", ",").split(",") if x.strip()]
    if len(parts) < 2:
        raise ValueError("custom sigmas: need at least 2 values")
    if len(parts) > 32:
        raise ValueError("custom sigmas: too many values (max 32)")
    try:
        vals = [float(x) for x in parts]
    except ValueError:
        raise ValueError("custom sigmas: all values must be numbers")
    if any(v < 0.0 or v > 1.0 for v in vals):
        raise ValueError("custom sigmas: all values must be in [0, 1]")
    for i in range(len(vals) - 1):
        if vals[i] <= vals[i + 1]:
            raise ValueError("custom sigmas: must be strictly decreasing")
    if vals[-1] > 0.01:
        raise ValueError("custom sigmas: last value must be ~0 (e.g. 0.0)")
    return ", ".join(f"{v:g}" for v in vals)


def _resolve_sigmas(preset: str, custom: str) -> str:
    if preset == "custom":
        return _validate_sigmas(custom)
    return SIGMA_PRESETS.get(preset, SIGMA_PRESETS["original"])


def _inject_refine_sigmas(workflow: dict[str, Any], sigma_str: str) -> None:
    node = workflow.get(NODE_REFINE_SIGMAS)
    if node is None:
        return
    inputs = node.get("inputs") or {}
    # KJNodes ManualSigmas input name is `sigmas_string`. Fall back to any
    # comma-stringy input if a future converter rename happens.
    if "sigmas_string" in inputs:
        inputs["sigmas_string"] = sigma_str
    else:
        for k, v in list(inputs.items()):
            if isinstance(v, str) and "," in v:
                inputs[k] = sigma_str
                break


def _redirect_consumers(
    workflow: dict[str, Any],
    old_ref: list,
    new_ref: list,
    exclude_node_ids: set[str] | None = None,
) -> int:
    """For every node input whose value == old_ref ([node_id, output_idx]),
    replace it with new_ref. Returns count of replacements.

    `exclude_node_ids` skips replacement INSIDE those nodes - critical when
    new_ref is itself a node that legitimately depends on old_ref (e.g. our
    MSR guide node has inputs pointing at LikenessGuide; redirecting those
    would create a self-reference cycle).
    """
    exclude = exclude_node_ids or set()
    n = 0
    for node_id, node in workflow.items():
        if node_id in exclude:
            continue
        ins = node.get("inputs") or {}
        for k, v in list(ins.items()):
            if isinstance(v, list) and len(v) == 2 and v == old_ref:
                ins[k] = list(new_ref)
                n += 1
    return n


def _inject_msr(
    workflow: dict[str, Any],
    width: int,
    height: int,
    output_frames: int,
    frame_count: int,
    guide_strength: float,
    msr_lora_strength: float,
    ref1_image_name: str,
    ref2_image_name: str | None,
    ref3_image_name: str | None,
    ref4_image_name: str | None,
    bg_image_name: str | None,
) -> None:
    """Patch the workflow to enable Multi-Subject Reference mode.

    Architecture:
    - LTXICLoRALoaderModelOnly loads the MSR ic-lora into the model chain
      BEFORE the rgthree power loader (installs ic-lora-specific
      reference_downscale_factor + model hooks; plain rgthree power loading
      does NOT install these hooks, just loads weights).
    - LiconMSR packs 1-4 refs + 1 background into a pseudo-video.
    - LTXAddVideoICLoRAGuide injects the pseudo-video as conditioning frames.
    - LTXVAddGuideMulti adds per-image positional anchors so the model gets
      per-image conditioning instead of one undifferentiated blob.
    - LTXVCropGuides strips the conditioning frames off the END before final
      VAE decode so the output is clean.
    - EmptyLTXVLatentVideo.length is extended by frame_count so the requested
      duration survives the MSR overhead.
    - LikenessGuide / LikenessAnchor / LatentAnchorAware are bypassed;
      identity in MSR mode comes entirely from ic-lora.
    """
    required = {
        "LikenessGuide": MSR_NODE_LIKENESS_GUIDE,
        "InplaceKJ-pass1": MSR_NODE_INPLACE_PASS1,
        "ConcatAV-pass1": MSR_NODE_CONCAT_PASS1,
        "SeparateAV-final": MSR_NODE_FINAL_SEPARATE,
        "VAEDecode-final": MSR_NODE_VAE_DECODE,
        "EmptyLatentVideo": MSR_NODE_EMPTY_LATENT,
    }
    missing = [f"{label}={nid}" for label, nid in required.items() if nid not in workflow]
    if missing:
        # Bail without changes if the expected node ids aren't present so
        # the error message is explicit rather than silent breakage.
        raise RuntimeError(f"MSR: required workflow nodes missing: {', '.join(missing)}")
    guide_node = workflow[MSR_NODE_LIKENESS_GUIDE]
    inplace_node = workflow[MSR_NODE_INPLACE_PASS1]
    concat_node = workflow[MSR_NODE_CONCAT_PASS1]
    separate_node = workflow[MSR_NODE_FINAL_SEPARATE]
    decode_node = workflow[MSR_NODE_VAE_DECODE]
    empty_latent_node = workflow[MSR_NODE_EMPTY_LATENT]

    guide_inputs = guide_node["inputs"]
    vae_ref = guide_inputs.get("vae")
    if vae_ref is None:
        raise RuntimeError("MSR: vae input missing on likeness guide; cannot inject")

    # Bypass the entire face/likeness/anchor identity stack - MSR is doing
    # identity work via the trained IC-LoRA.
    guide_inputs["strength"] = 0.0
    guide_inputs["face_detect"] = "none"
    guide_inputs["face_bbox_within_reference"] = ""
    guide_inputs["reference_mask_mode"] = "bbox_only"
    guide_inputs["emit_latent"] = "passthrough"

    anchor_node = workflow.get(MSR_NODE_LIKENESS_ANCHOR)
    if anchor_node:
        anchor_node["inputs"]["strength"] = 0.0
        anchor_node["inputs"]["bypass"] = True

    latent_anchor_node = workflow.get(MSR_NODE_LATENT_ANCHOR)
    if latent_anchor_node:
        latent_anchor_node["inputs"]["strength"] = 0.0
        latent_anchor_node["inputs"]["bypass"] = True

    # Extend EmptyLTXVLatentVideo.length to absorb MSR overhead.
    # LTXAddVideoICLoRAGuide consumes latent frames (assertion: conditioning
    # fits within latent_length). 41 image frames of MSR = ~6 latent frames.
    # Without extending, the requested 4s gets truncated to ~1s post-crop.
    # Length replaced with a literal int; the visual workflow wires length
    # through a slider/SetNode chain that _set_slider modifies, so writing a
    # literal severs that chain. Total = output_frames + frame_count, rounded
    # up to nearest 8n+1.
    raw_total = max(9, int(output_frames) + int(frame_count))
    n_block = (raw_total - 1 + 7) // 8  # ceil((raw_total-1) / 8)
    extended_length = max(9, n_block * 8 + 1)
    empty_latent_node["inputs"]["length"] = int(extended_length)

    # Add 4 new LoadImage nodes for the additional MSR refs + background.
    new_load_nodes: dict[str, str] = {}
    for new_id, fname in (
        (MSR_NEW_REF_2, ref2_image_name),
        (MSR_NEW_REF_3, ref3_image_name),
        (MSR_NEW_REF_4, ref4_image_name),
        (MSR_NEW_BG, bg_image_name),
    ):
        if fname:
            workflow[new_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": fname, "upload": "image"},
            }
            new_load_nodes[new_id] = fname

    # If no background was provided, MSR's `background` input is required by
    # the node. Use ref1 as background fallback.
    bg_source: list = [MSR_NEW_BG, 0] if MSR_NEW_BG in new_load_nodes else [NODE_LOAD_IMAGE, 0]

    # LiconMSR: packs refs into pseudo-video.
    msr_inputs: dict[str, Any] = {
        "width": int(width),
        "height": int(height),
        "frame_count": int(frame_count),
        "1": [NODE_LOAD_IMAGE, 0],
        "background": bg_source,
    }
    if MSR_NEW_REF_2 in new_load_nodes:
        msr_inputs["2"] = [MSR_NEW_REF_2, 0]
    if MSR_NEW_REF_3 in new_load_nodes:
        msr_inputs["3"] = [MSR_NEW_REF_3, 0]
    if MSR_NEW_REF_4 in new_load_nodes:
        msr_inputs["4"] = [MSR_NEW_REF_4, 0]
    workflow[MSR_NEW_PSEUDO_VIDEO] = {
        "class_type": "LiconMSR",
        "inputs": msr_inputs,
    }

    # LTXAddVideoICLoRAGuide: pseudo-video → conditioning frames inside latent.
    workflow[MSR_NEW_GUIDE] = {
        "class_type": "LTXAddVideoICLoRAGuide",
        "inputs": {
            "positive": [MSR_NODE_LIKENESS_GUIDE, 0],
            "negative": [MSR_NODE_LIKENESS_GUIDE, 1],
            "vae": list(vae_ref),
            "latent": [MSR_NODE_LIKENESS_GUIDE, 2],
            "image": [MSR_NEW_PSEUDO_VIDEO, 0],
            "frame_idx": 0,
            "strength": float(guide_strength),
            "latent_downscale_factor": 1.0,
            "crop": "center",
            "use_tiled_encode": False,
            "tile_size": 256,
            "tile_overlap": 64,
        },
    }

    # LTXVAddGuideMulti: places each reference image at its own frame_idx
    # with its own strength on top of the pseudo-video conditioning, so the
    # model gets per-image positional anchoring instead of one undifferentiated
    # blob. API form: top-level `num_guides` is a string count ("1"-"20"); per-
    # guide inputs are namespaced as `num_guides.image_N` /
    # `num_guides.frame_idx_N` / `num_guides.strength_N`.
    guide_multi_images: list[list] = [[NODE_LOAD_IMAGE, 0]]  # ref1 always
    if MSR_NEW_REF_2 in new_load_nodes:
        guide_multi_images.append([MSR_NEW_REF_2, 0])
    if MSR_NEW_REF_3 in new_load_nodes:
        guide_multi_images.append([MSR_NEW_REF_3, 0])
    if MSR_NEW_REF_4 in new_load_nodes:
        guide_multi_images.append([MSR_NEW_REF_4, 0])
    if MSR_NEW_BG in new_load_nodes:
        guide_multi_images.append([MSR_NEW_BG, 0])

    multi_count = len(guide_multi_images)
    multi_inputs: dict[str, Any] = {
        "positive": [MSR_NEW_GUIDE, 0],
        "negative": [MSR_NEW_GUIDE, 1],
        "vae": list(vae_ref),
        "latent": [MSR_NEW_GUIDE, 2],
        # DynamicCombo: top-level value is the count as a string; per-guide
        # widgets/inputs are namespaced with the `num_guides.` prefix.
        "num_guides": str(multi_count),
    }
    per_guide_strength = max(0.05, float(guide_strength))
    for i, img_ref in enumerate(guide_multi_images, start=1):
        multi_inputs[f"num_guides.image_{i}"] = img_ref
        multi_inputs[f"num_guides.frame_idx_{i}"] = 0
        multi_inputs[f"num_guides.strength_{i}"] = per_guide_strength
    workflow[MSR_NEW_GUIDE_MULTI] = {
        "class_type": "LTXVAddGuideMulti",
        "inputs": multi_inputs,
    }

    # LTXVCropGuides: strips MSR conditioning frames from latent before final
    # decode. positive/negative come from LTXAddVideoICLoRAGuide DIRECTLY (not
    # through LTXVAddGuideMulti) - Multi's conditioning has multi-layered guide
    # metadata that confuses the crop logic. Only Multi's LATENT output is
    # consumed downstream (into ConcatAV.video_latent).
    workflow[MSR_NEW_CROP] = {
        "class_type": "LTXVCropGuides",
        "inputs": {
            "positive": [MSR_NEW_GUIDE, 0],
            "negative": [MSR_NEW_GUIDE, 1],
            "latent": [MSR_NODE_FINAL_SEPARATE, 0],
        },
    }

    # Rewire LikenessGuide.positive/negative consumers (CFGGuider, STGGuider)
    # to LTXAddVideoICLoRAGuide DIRECTLY (not through LTXVAddGuideMulti).
    # LTXVAddGuideMulti.positive/negative outputs are unused; only its latent
    # is consumed (by ConcatAV).
    # CRITICAL: exclude MSR_NEW_GUIDE from the redirect since it legitimately
    # consumes LikenessGuide outputs; without exclusion the redirect creates
    # a self-referencing cycle (msr_guide.positive = [msr_guide, 0]) and
    # comfy silently skips the conditioning chain.
    redirect_exclude = {MSR_NEW_GUIDE}
    _redirect_consumers(workflow,
                        [MSR_NODE_LIKENESS_GUIDE, 0],
                        [MSR_NEW_GUIDE, 0],
                        exclude_node_ids=redirect_exclude)
    _redirect_consumers(workflow,
                        [MSR_NODE_LIKENESS_GUIDE, 1],
                        [MSR_NEW_GUIDE, 1],
                        exclude_node_ids=redirect_exclude)
    # ConcatAV.video_latent receives LTXVAddGuideMulti's latent (has both the
    # MSR pseudo-video AND per-image keyframes appended).
    concat_node["inputs"]["video_latent"] = [MSR_NEW_GUIDE_MULTI, 2]
    # VAEDecode samples come from the crop guides output (latent slot 2).
    decode_node["inputs"]["samples"] = [MSR_NEW_CROP, 2]

    # Install MSR via LTXICLoRALoaderModelOnly, NOT rgthree. Plain Power Lora
    # Loader only loads weights; LTXICLoRALoaderModelOnly additionally extracts
    # reference_downscale_factor from safetensors metadata and installs the
    # IC-LoRA-specific model patches that enable correct inference behavior.
    # New chain: ckpt -> LTXICLoRALoaderModelOnly -> Power Lora Loader ->
    # CFGGuider/STGGuider. The IC-LoRA loader is spliced BEFORE the rgthree
    # loader by stealing rgthree's `model` upstream connection.
    power_loader = workflow.get(NODE_POWER_LORA)
    if msr_lora_strength > 0 and power_loader is not None:
        # Clear any stale lora_msr entry from prior versions.
        power_loader["inputs"].pop("lora_msr", None)
        upstream_model = power_loader["inputs"].get("model")
        if upstream_model is None:
            raise RuntimeError(
                "MSR: power loader has no upstream model connection; "
                "cannot splice IC-LoRA loader."
            )
        workflow[MSR_NEW_ICLORA_LOADER] = {
            "class_type": "LTXICLoRALoaderModelOnly",
            "inputs": {
                "model": list(upstream_model) if isinstance(upstream_model, list) else upstream_model,
                "lora_name": MSR_LORA_FILENAME,
                "strength_model": float(msr_lora_strength),
            },
        }
        power_loader["inputs"]["model"] = [MSR_NEW_ICLORA_LOADER, 0]


_RELAY_SEGMENT_RE = re.compile(
    r'^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*:\s*(.+?)\s*$'
)


def _prompt_relay_smart_prompt(text: str, duration_seconds: float) -> str:
    """Convert legacy second ranges to PromptRelaySmartEncode syntax.

    If every non-empty line matches `start-end: text`, convert it to official
    pipe syntax with `[start-end]` tags. Otherwise pass text through so the
    plugin can parse its native pipe/block smart formats.
    """
    if not text or not text.strip():
        return ""
    out: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _RELAY_SEGMENT_RE.match(line)
        if not m:
            return text.strip()
        try:
            start = float(m.group(1))
            end = float(m.group(2))
        except (TypeError, ValueError):
            return text.strip()
        body = m.group(3).strip()
        if not body or end <= start or start < 0:
            return text.strip()
        if end > duration_seconds + 0.01:  # 10ms tolerance
            return text.strip()
        out.append(f"{body} [{start:g}-{end:g}]")
    return " | ".join(out)


def _inject_prompt_relay(
    workflow: dict[str, Any],
    smart_prompt: str,
    global_prompt: str,
    epsilon: float = 0.001,
) -> bool:
    """Splice a PromptRelayEncode node between Power Lora Loader and its
    downstream consumers, and route its conditioning output into the
    LTXVConditioning node's positive input.

    Returns True on successful injection, False if any required upstream
    node is missing (caller falls back to single-prompt behavior).
    """
    if not smart_prompt or not smart_prompt.strip():
        return False
    required = (NODE_POWER_LORA, NODE_TEXT_ENCODER, MSR_NODE_EMPTY_LATENT,
                NODE_LTXV_CONDITIONING, NODE_POSITIVE)
    if not all(nid in workflow for nid in required):
        return False
    power_loader = workflow[NODE_POWER_LORA]
    upstream_model = power_loader["inputs"].get("model")
    if upstream_model is None:
        return False

    workflow[RELAY_NEW_NODE] = {
        "class_type": "PromptRelaySmartEncode",
        "inputs": {
            "model": list(upstream_model) if isinstance(upstream_model, list) else upstream_model,
            "clip": [NODE_TEXT_ENCODER, 0],
            "latent": [MSR_NODE_EMPTY_LATENT, 0],
            "global_prompt": str(global_prompt or ""),
            "smart_prompt": str(smart_prompt or ""),
            "normalize_by_tokens": False,
            "epsilon": float(epsilon),
        },
    }
    # Reroute Power Lora Loader's model output through the relay node so all
    # downstream model consumers get the attention-patched model.
    power_loader["inputs"]["model"] = [RELAY_NEW_NODE, 0]
    # Replace LTXVConditioning's positive input with the relay's conditioning
    # output. Negative path stays on the existing CLIPTextEncode node.
    cond_inputs = workflow[NODE_LTXV_CONDITIONING]["inputs"]
    cond_inputs["positive"] = [RELAY_NEW_NODE, 1]
    return True


_SCENE_CHAIN_HEADER_RE = re.compile(r"^\s*scene\s+\d+\s*:\s*$", re.IGNORECASE)


def _parse_scene_chain_scenes(text: str, max_scenes: int = 2) -> list[str]:
    if not text or not text.strip():
        return []
    scenes: list[str] = []
    current: list[str] = []
    seen_header = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if _SCENE_CHAIN_HEADER_RE.match(line):
            if current:
                body = " ".join(current).strip()
                if body:
                    scenes.append(body)
                current = []
            seen_header = True
            continue
        if seen_header and line:
            current.append(line)
    if current:
        body = " ".join(current).strip()
        if body:
            scenes.append(body)
    limit = max(1, int(max_scenes or 1))
    return scenes[:limit]


def _join_scene_prompt(global_prompt: str, scene_prompt: str) -> str:
    global_prompt = str(global_prompt or "").strip()
    scene_prompt = str(scene_prompt or "").strip()
    if not global_prompt:
        return scene_prompt
    if not scene_prompt:
        return global_prompt
    sep = " " if global_prompt[-1:] in ".!?,\"'" else ", "
    return f"{global_prompt}{sep}{scene_prompt}"


def _scene_chain_frames(total_frames: int, scene_count: int, fps: int = 24) -> int:
    scene_count = max(1, int(scene_count or 1))
    total_seconds = max(1.0 / fps, (int(total_frames) - 1) / float(fps))
    return _safe_frames(total_seconds / scene_count, fps=fps)


def _inject_scene_chain(
    workflow: dict[str, Any],
    *,
    scenes: list[str],
    global_prompt: str,
    total_frames: int,
    frame_overlap: int = 8,
    mid_scene_guide: bool = True,
    mid_scene_guide_strength: float = 0.25,
) -> bool:
    if len(scenes) < 2:
        return False
    required = (
        NODE_TEXT_ENCODER, NODE_NEGATIVE, NODE_LTXV_CONDITIONING,
        NODE_LIKENESS_GUIDE, NODE_LIKENESS_ANCHOR, NODE_VIDEO_VAE,
        NODE_FIRST_PASS_SAMPLER_SELECT, NODE_FIRST_PASS_SIGMAS,
        NODE_FIRST_PASS_LATENT, NODE_SEED, NODE_FINAL_SEPARATE,
    )
    if not all(nid in workflow for nid in required):
        return False

    frame_rate_ref = workflow[NODE_LTXV_CONDITIONING]["inputs"].get("frame_rate")
    negative_ref = workflow[NODE_NEGATIVE]["inputs"].get("text")
    scene_refs: list[list[Any]] = []
    for index, scene in enumerate(scenes):
        clip_node = f"{SCENE_CHAIN_NODE_PREFIX}_clip_{index}"
        conditioning_node = f"{SCENE_CHAIN_NODE_PREFIX}_conditioning_{index}"
        workflow[clip_node] = {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": [NODE_TEXT_ENCODER, 0],
                "text": _join_scene_prompt(global_prompt, scene),
            },
        }
        workflow[conditioning_node] = {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": [clip_node, 0],
                "negative": [NODE_NEGATIVE, 0],
                "frame_rate": list(frame_rate_ref) if isinstance(frame_rate_ref, list) else frame_rate_ref,
            },
        }
        scene_refs.append([conditioning_node, 0])

    combined_ref = scene_refs[0]
    for index, scene_ref in enumerate(scene_refs[1:], start=1):
        combine_node = f"{SCENE_CHAIN_NODE_PREFIX}_combine_{index}"
        workflow[combine_node] = {
            "class_type": "ConditioningCombine",
            "inputs": {
                "conditioning_1": combined_ref,
                "conditioning_2": scene_ref,
            },
        }
        combined_ref = [combine_node, 0]

    workflow[NODE_LIKENESS_GUIDE]["inputs"]["positive"] = combined_ref

    per_scene_frames = _scene_chain_frames(int(total_frames), len(scenes))
    max_overlap = max(0, per_scene_frames - 9)
    resolved_overlap = max(0, min(int(frame_overlap), max_overlap))
    _set_slider(workflow, NODE_LENGTH, max(1, per_scene_frames - 1))

    workflow[SCENE_CHAIN_NEW_NODE] = {
        "class_type": "FunPackLTXAVSceneChainSampler",
        "inputs": {
            "model": [NODE_LIKENESS_ANCHOR, 0],
            "vae": [NODE_VIDEO_VAE, 0],
            "positive": [NODE_LIKENESS_GUIDE, 0],
            "negative": [NODE_LIKENESS_GUIDE, 1],
            "sampler": [NODE_FIRST_PASS_SAMPLER_SELECT, 0],
            "sigmas": [NODE_FIRST_PASS_SIGMAS, 0],
            "seed": [NODE_SEED, 0],
            "latent_template": [NODE_FIRST_PASS_LATENT, 0],
            "num_frames_per_scene": int(per_scene_frames),
            "frame_overlap": int(resolved_overlap),
            "cfg": 1.0,
            "max_scenes": len(scenes),
            "use_same_seed": False,
            "carry_i2v_guides": False,
            "mid_scene_guide": bool(mid_scene_guide),
            "mid_scene_guide_strength": float(mid_scene_guide_strength),
            "embed_guidance": False,
            "embed_guidance_strength": 0.02,
            "transition_duration": 0,
        },
    }
    workflow[NODE_FINAL_SEPARATE]["inputs"]["av_latent"] = [SCENE_CHAIN_NEW_NODE, 0]
    return True


def _inject_kv_conditioning(workflow: dict[str, Any], strength: float = 1.0) -> bool:
    """Splice a FunPackKVApply node between Power Lora Loader and its
    downstream model consumers. The wrapper invokes FunPack's
    build_enhancements which patches the model with K/V hidden state
    injection from the i2v reference latent. The strength input scales
    every hook firing through a monkey-patch on _sigma_gated_strength.

    Returns True on success, False if required upstream nodes are absent.
    """
    required = (NODE_POWER_LORA, NODE_I2V_REF_LATENT, NODE_POSITIVE)
    if not all(nid in workflow for nid in required):
        return False
    power_loader = workflow[NODE_POWER_LORA]
    upstream_model = power_loader["inputs"].get("model")
    if upstream_model is None:
        return False

    workflow[KV_NEW_NODE] = {
        "class_type": "FunPackKVApply",
        "inputs": {
            "model": list(upstream_model) if isinstance(upstream_model, list) else upstream_model,
            "latent": [NODE_I2V_REF_LATENT, 0],
            "conditioning": [NODE_POSITIVE, 0],
            "strength": float(strength),
            "temporal_style": "natural",
        },
    }
    # Route the patched model output back into power_loader's downstream
    # consumers - downstream lora chain + guiders see the K/V-patched model.
    power_loader["inputs"]["model"] = [KV_NEW_NODE, 0]
    return True


def _inject_audio_reference(
    workflow: dict[str, Any],
    audio_filename: str,
    guidance_scale: float = 3.0,
    stem_sep: bool = False,
    normalize_audio: bool = True,
) -> bool:
    """Splice an LTXVReferenceAudio node between Power Lora Loader and its
    downstream model consumers, also patching the positive/negative
    conditioning chain. The node encodes the ref audio via the existing
    LTXVAudioVAELoader (617), patches model with identity guidance, and
    routes through patched conditioning.

    Reference audio is capped to 10s. When stem_sep=True we trim before
    MelBandRoFormer, then normalize the separated vocals before encoding.

    Returns True on success, False if required upstream nodes are absent.
    """
    required = (NODE_POWER_LORA, NODE_POSITIVE, NODE_NEGATIVE, NODE_AUDIO_VAE_LOADER)
    if not all(nid in workflow for nid in required):
        return False
    power_loader = workflow[NODE_POWER_LORA]
    upstream_model = power_loader["inputs"].get("model")
    if upstream_model is None:
        return False

    # LoadAudio reads from comfy's INPUT dir by filename.
    workflow[AUDIO_REF_NEW_LOAD] = {
        "class_type": "LoadAudio",
        "inputs": {"audio": audio_filename},
    }
    ref_audio_source: list = [AUDIO_REF_NEW_LOAD, 0]

    if stem_sep:
        workflow[AUDIO_REF_NEW_TRIM] = {
            "class_type": "AudioRefPrep",
            "inputs": {
                "audio": ref_audio_source,
                "normalize": False,
                "max_seconds": 10.0,
                "target_peak_db": -3.0,
                "max_gain_db": 24.0,
            },
        }
        # MelBandRoFormer separates vocals from instruments.
        # Model loaded from models/diffusion_models/.
        workflow[AUDIO_REF_NEW_MEL_LOADER] = {
            "class_type": "MelBandRoFormerModelLoader",
            "inputs": {"model_name": "MelBandRoformer_fp16.safetensors"},
        }
        workflow[AUDIO_REF_NEW_MEL_SAMPLER] = {
            "class_type": "MelBandRoFormerSampler",
            "inputs": {
                "model": [AUDIO_REF_NEW_MEL_LOADER, 0],
                "audio": [AUDIO_REF_NEW_TRIM, 0],
            },
        }
        ref_audio_source = [AUDIO_REF_NEW_MEL_SAMPLER, 0]  # vocals

    workflow[AUDIO_REF_NEW_PREP] = {
        "class_type": "AudioRefPrep",
        "inputs": {
            "audio": ref_audio_source,
            "normalize": bool(normalize_audio),
            "max_seconds": 10.0,
            "target_peak_db": -3.0,
            "max_gain_db": 24.0,
        },
    }
    ref_audio_source = [AUDIO_REF_NEW_PREP, 0]

    # LTXVReferenceAudio patches model + conditioning.
    workflow[AUDIO_REF_NEW_NODE] = {
        "class_type": "LTXVReferenceAudio",
        "inputs": {
            "model": list(upstream_model) if isinstance(upstream_model, list) else upstream_model,
            "positive": [NODE_POSITIVE, 0],
            "negative": [NODE_NEGATIVE, 0],
            "reference_audio": ref_audio_source,
            "audio_vae": [NODE_AUDIO_VAE_LOADER, 0],
            "identity_guidance_scale": float(guidance_scale),
            "start_percent": 0.0,
            "end_percent": 1.0,
        },
    }

    # Route Power Lora's model through the audio-ref-patched model.
    power_loader["inputs"]["model"] = [AUDIO_REF_NEW_NODE, 0]

    # Reroute downstream conditioning consumers through patched outputs.
    # Slot 1 = patched positive, slot 2 = patched negative.
    # Exclude AUDIO_REF_NEW_NODE itself (self-reference) and KV_NEW_NODE
    # (KV reads raw POSITIVE as context-only signal; redirecting would
    # create a cycle since AUDIO_REF.model = [KV_NEW, 0]).
    exclude = {AUDIO_REF_NEW_NODE}
    if KV_NEW_NODE in workflow:
        exclude.add(KV_NEW_NODE)
    _redirect_consumers(
        workflow, [NODE_POSITIVE, 0], [AUDIO_REF_NEW_NODE, 1],
        exclude_node_ids=exclude,
    )
    _redirect_consumers(
        workflow, [NODE_NEGATIVE, 0], [AUDIO_REF_NEW_NODE, 2],
        exclude_node_ids=exclude,
    )
    return True


def _safe_frames(seconds: float, fps: int = 24) -> int:
    frames = max(9, int(seconds * fps) + 1)
    return ((frames - 1 + 7) // 8) * 8 + 1


_RES_MIN_MP = 1.0
_RES_MAX_MP = 1.2


def _fit_dimensions(
    image: Image.Image,
    max_width: int,
    max_height: int,
    snap: int = 64,
    target_mp: float = 1.15,
    custom_res: bool = False,
) -> tuple[int, int]:
    s = max(32, int(snap))
    if custom_res:
        scale = min(max_width / image.width, max_height / image.height)
        width = max(s, round(image.width * scale / s) * s)
        height = max(s, round(image.height * scale / s) * s)
    else:
        mp = max(0.1, float(target_mp)) * 1_000_000
        ar = image.width / image.height
        height = max(s, round((mp / ar) ** 0.5 / s) * s)
        width = max(s, round((mp * ar) ** 0.5 / s) * s)
        if width * height < _RES_MIN_MP * 1_000_000:
            opt_h = (width * (height + s), height + s, width)
            opt_w = (height * (width + s), height, width + s)
            best = min(opt_h, opt_w, key=lambda x: x[0])
            height, width = best[1], best[2]
        if width * height > _RES_MAX_MP * 1_000_000:
            opt_h = (width * (height - s), height - s, width)
            opt_w = (height * (width - s), height, width - s)
            best = max(opt_h, opt_w, key=lambda x: x[0])
            height, width = best[1], best[2]
    return width, height


def _execute_workflow(workflow: dict[str, Any]) -> str:
    import execution
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    executor = execution.PromptExecutor(
        server_instance,
        cache_type=execution.CacheType.RAM_PRESSURE,
        cache_args={"lru": 0, "ram": 2.0, "ram_inactive": 8.0},
    )
    prompt_id = str(uuid.uuid4())
    executor.execute(workflow, prompt_id, extra_data={}, execute_outputs=[NODE_OUTPUT])
    if not executor.success:
        raise RuntimeError(str(executor.status_messages[-1] if executor.status_messages else "comfy execution failed"))

    paths: list[pathlib.Path] = []
    for output in executor.history_result.get("outputs", {}).values():
        for items in output.values():
            if not isinstance(items, list):
                continue
            for item in items:
                filename = item.get("filename") if isinstance(item, dict) else None
                if not filename:
                    continue
                subfolder = item.get("subfolder", "")
                kind = item.get("type", "output")
                base = OUTPUT if kind == "output" else COMFY / kind
                candidate = base / subfolder / filename if subfolder else base / filename
                if candidate.exists():
                    paths.append(candidate)
    if not paths:
        files = [pathlib.Path(p) for p in glob.glob(str(OUTPUT / "**" / "*.mp4"), recursive=True)]
        paths = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    if not paths:
        raise RuntimeError("comfy finished without an output video")
    return str(paths[0])


def _prepare_runtime(progress: gr.Progress | None = None) -> None:
    _ensure_comfy()
    _ensure_models(progress)
    _init_comfy_nodes()


def get_gpu_duration(
    image_path: str,
    prompt: str,
    negative_prompt: str,
    preset: str,
    seconds: float,
    max_width: int,
    max_height: int,
    mode: str,
    face_bbox: str,
    likeness_strength: float,
    likeness_anchor_strength: float,
    latent_anchor_strength: float,
    first_frame_strength: float,
    seed: int,
    randomize_seed: bool,
    gen_budget: float = 0,
    target_mp: float = 1.15,
    snap_multiple: int = 64,
    custom_res_enabled: bool = False,
    sulphur_lora_strength: float = 0.15,
    sulphur_v1_lora_strength: float = 0.15,
    vbvr_lora_strength: float = 0.5,
    dreamly_lora_strength: float = 0.6,
    synth_lora_strength: float = 0.0,
    plora_lora_strength: float = 0.0,
    singularity_lora_strength: float = 0.3,
    omninft_lora_strength: float = 0.8,
    omninft_bf16_lora_strength: float = 0.0,
    better_motion_lora_strength: float = 0.0,
    physics_v2_lora_strength: float = 0.0,
    hardcut_lora_strength: float = 0.0,
    transition_lora_strength: float = 0.15,
    slot7_lora_strength: float = 0.0,
    sulphur_audio_strength: float = 0.15,
    sulphur_v1_audio_strength: float = 0.15,
    vbvr_audio_strength: float = 0.5,
    dreamly_audio_strength: float = 0.6,
    synth_audio_strength: float = 0.0,
    plora_audio_strength: float = 0.0,
    singularity_audio_strength: float = 0.3,
    omninft_audio_strength: float = 0.8,
    omninft_bf16_audio_strength: float = 0.0,
    better_motion_audio_strength: float = 0.0,
    physics_v2_audio_strength: float = 0.0,
    hardcut_audio_strength: float = 0.0,
    transition_audio_strength: float = 0.0,
    slot7_audio_strength: float = 0.0,
    slot1_lora_file: str = HARDCUT_LORA_FILENAME,
    slot2_lora_file: str = SYNTH_LORA_FILENAME,
    slot3_lora_file: str = PLORA_LORA_FILENAME,
    slot4_lora_file: str = OMNINFT_BF16_LORA_FILENAME,
    slot5_lora_file: str = BETTER_MOTION_LORA_FILENAME,
    slot6_lora_file: str = PHYSICS_V2_LORA_FILENAME,
    slot7_lora_file: str = LORA_NONE,
    cache_at_step: int = 0,
    cache_warmup: int = 400,
    energy_threshold: float = 0.3,
    anchor_similarity_threshold: float = 0.3,
    sigma_string: str = _SIGMA_TUNED,
    input_mode: str = "single image (i2v)",
    msr_ref2: str | None = None,
    msr_ref3: str | None = None,
    msr_ref4: str | None = None,
    msr_background: str | None = None,
    msr_frame_count: int = 41,
    msr_guide_strength: float = 1.0,
    msr_lora_strength: float = 0.7,
    prompt_relay_enabled: bool = False,
    prompt_segments: str = "",
    scene_chain_enabled: bool = False,
    scene_chain_prompt: str = "",
    scene_chain_max_scenes: int = 2,
    scene_chain_frame_overlap: int = 8,
    scene_chain_mid_guide: bool = True,
    scene_chain_mid_guide_strength: float = 0.25,
    kv_enabled: bool = False,
    kv_strength: float = 1.0,
    audio_ref_enabled: bool = False,
    audio_ref_file: str | None = None,
    audio_ref_guidance_scale: float = 3.0,
    audio_ref_stem_sep: bool = False,
    audio_ref_normalize: bool = True,
    custom1_lora_strength: float = 0.0,
    custom2_lora_strength: float = 0.0,
    custom3_lora_strength: float = 0.0,
    custom4_lora_strength: float = 0.0,
    custom5_lora_strength: float = 0.0,
    custom6_lora_strength: float = 0.0,
    custom1_audio_strength: float = 0.0,
    custom2_audio_strength: float = 0.0,
    custom3_audio_strength: float = 0.0,
    custom4_audio_strength: float = 0.0,
    custom5_audio_strength: float = 0.0,
    custom6_audio_strength: float = 0.0,
    custom1_lora_file: str = LORA_NONE,
    custom2_lora_file: str = LORA_NONE,
    custom3_lora_file: str = LORA_NONE,
    custom4_lora_file: str = LORA_NONE,
    custom5_lora_file: str = LORA_NONE,
    custom6_lora_file: str = LORA_NONE,
    progress: gr.Progress | None = None,
) -> int:
    # Manual override: gen_budget > 0 forces an exact GPU budget.
    if gen_budget and int(gen_budget) > 0:
        return max(MIN_GPU_SECONDS, min(MAX_GPU_SECONDS, int(gen_budget)))
    frames = _safe_frames(float(seconds))
    if custom_res_enabled:
        pixels = max(64, int(max_width)) * max(64, int(max_height))
    else:
        pixels = max(64 * 64, int(max(0.1, float(target_mp)) * 1_000_000))
    base_work = _safe_frames(1.0) * 512 * 640
    work = frames * pixels / base_work
    mode_cost = 1.10 if mode != "anchor only" else 1.0
    if input_mode == "multi-reference (MSR)":
        mode_cost *= 1.10
    # Two regimes: tight (30+5*work) lets default 4s fit the 120s/day free
    # ZeroGPU allowance; anything longer falls back to the older wider
    # formula (45+8*work) that's proven to complete on long gens.
    tight = 30 + int(5.0 * work * mode_cost)
    if tight <= 120:
        estimate = tight
    else:
        estimate = MIN_GPU_SECONDS + int(8.0 * work * mode_cost)
    return max(MIN_GPU_SECONDS, min(MAX_GPU_SECONDS, estimate))


@spaces.GPU(duration=get_gpu_duration)
def generate(
    image_path: str,
    prompt: str,
    negative_prompt: str,
    preset: str,
    seconds: float,
    max_width: int,
    max_height: int,
    mode: str,
    face_bbox: str,
    likeness_strength: float,
    likeness_anchor_strength: float,
    latent_anchor_strength: float,
    first_frame_strength: float,
    seed: int,
    randomize_seed: bool,
    gen_budget: float = 0,
    target_mp: float = 1.15,
    snap_multiple: int = 64,
    custom_res_enabled: bool = False,
    sulphur_lora_strength: float = 0.15,
    sulphur_v1_lora_strength: float = 0.15,
    vbvr_lora_strength: float = 0.5,
    dreamly_lora_strength: float = 0.6,
    synth_lora_strength: float = 0.0,
    plora_lora_strength: float = 0.0,
    singularity_lora_strength: float = 0.3,
    omninft_lora_strength: float = 0.8,
    omninft_bf16_lora_strength: float = 0.0,
    better_motion_lora_strength: float = 0.0,
    physics_v2_lora_strength: float = 0.0,
    hardcut_lora_strength: float = 0.0,
    transition_lora_strength: float = 0.15,
    slot7_lora_strength: float = 0.0,
    sulphur_audio_strength: float = 0.15,
    sulphur_v1_audio_strength: float = 0.15,
    vbvr_audio_strength: float = 0.5,
    dreamly_audio_strength: float = 0.6,
    synth_audio_strength: float = 0.0,
    plora_audio_strength: float = 0.0,
    singularity_audio_strength: float = 0.3,
    omninft_audio_strength: float = 0.8,
    omninft_bf16_audio_strength: float = 0.0,
    better_motion_audio_strength: float = 0.0,
    physics_v2_audio_strength: float = 0.0,
    hardcut_audio_strength: float = 0.0,
    transition_audio_strength: float = 0.0,
    slot7_audio_strength: float = 0.0,
    slot1_lora_file: str = HARDCUT_LORA_FILENAME,
    slot2_lora_file: str = SYNTH_LORA_FILENAME,
    slot3_lora_file: str = PLORA_LORA_FILENAME,
    slot4_lora_file: str = OMNINFT_BF16_LORA_FILENAME,
    slot5_lora_file: str = BETTER_MOTION_LORA_FILENAME,
    slot6_lora_file: str = PHYSICS_V2_LORA_FILENAME,
    slot7_lora_file: str = LORA_NONE,
    cache_at_step: int = 0,
    cache_warmup: int = 400,
    energy_threshold: float = 0.3,
    anchor_similarity_threshold: float = 0.3,
    sigma_string: str = _SIGMA_TUNED,
    input_mode: str = "single image (i2v)",
    msr_ref2: str | None = None,
    msr_ref3: str | None = None,
    msr_ref4: str | None = None,
    msr_background: str | None = None,
    msr_frame_count: int = 41,
    msr_guide_strength: float = 1.0,
    msr_lora_strength: float = 0.7,
    prompt_relay_enabled: bool = False,
    prompt_segments: str = "",
    scene_chain_enabled: bool = False,
    scene_chain_prompt: str = "",
    scene_chain_max_scenes: int = 2,
    scene_chain_frame_overlap: int = 8,
    scene_chain_mid_guide: bool = True,
    scene_chain_mid_guide_strength: float = 0.25,
    kv_enabled: bool = False,
    kv_strength: float = 1.0,
    audio_ref_enabled: bool = False,
    audio_ref_file: str | None = None,
    audio_ref_guidance_scale: float = 3.0,
    audio_ref_stem_sep: bool = False,
    audio_ref_normalize: bool = True,
    custom1_lora_strength: float = 0.0,
    custom2_lora_strength: float = 0.0,
    custom3_lora_strength: float = 0.0,
    custom4_lora_strength: float = 0.0,
    custom5_lora_strength: float = 0.0,
    custom6_lora_strength: float = 0.0,
    custom1_audio_strength: float = 0.0,
    custom2_audio_strength: float = 0.0,
    custom3_audio_strength: float = 0.0,
    custom4_audio_strength: float = 0.0,
    custom5_audio_strength: float = 0.0,
    custom6_audio_strength: float = 0.0,
    custom1_lora_file: str = LORA_NONE,
    custom2_lora_file: str = LORA_NONE,
    custom3_lora_file: str = LORA_NONE,
    custom4_lora_file: str = LORA_NONE,
    custom5_lora_file: str = LORA_NONE,
    custom6_lora_file: str = LORA_NONE,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str, str, int]:
    seed_value = random.randint(0, 2**32 - 1) if randomize_seed or seed < 0 else int(seed)
    msr_enabled = input_mode == "multi-reference (MSR)"
    msr_original = input_mode == "multi-reference (original)"
    any_msr = msr_enabled or msr_original
    try:
        if not image_path:
            raise ValueError("upload reference 1 first" if any_msr else "upload an image first")
        if not prompt.strip():
            raise ValueError("prompt is empty")
        progress(0.0, desc="preparing comfy")
        _prepare_runtime(progress)

        image = Image.open(image_path).convert("RGB")
        width, height = _fit_dimensions(
            image, int(max_width), int(max_height),
            snap=int(snap_multiple), target_mp=float(target_mp),
            custom_res=bool(custom_res_enabled),
        )
        frames = _safe_frames(float(seconds))

        image_name = f"input_{uuid.uuid4().hex[:10]}.png"
        image.save(INPUT / image_name, format="PNG")

        def _save_ref(path: str | None, label: str) -> str | None:
            if not path:
                return None
            try:
                p = pathlib.Path(path)
                if not p.exists():
                    return None
                ref_img = Image.open(path).convert("RGB").resize((width, height), Image.LANCZOS)
                name = f"input_{label}_{uuid.uuid4().hex[:10]}.png"
                ref_img.save(INPUT / name, format="PNG")
                return name
            except Exception as e:
                print(f"[msr] failed to save {label} ({path}): {e}", flush=True)
                return None

        msr_ref2_name = _save_ref(msr_ref2, "ref2") if any_msr else None
        msr_ref3_name = _save_ref(msr_ref3, "ref3") if msr_enabled else None
        msr_ref4_name = _save_ref(msr_ref4, "ref4") if msr_enabled else None
        msr_bg_name = _save_ref(msr_background, "bg") if any_msr else None

        # Copy audio reference into comfy's INPUT dir so LoadAudio can find it.
        audio_ref_name: str | None = None
        if audio_ref_enabled and audio_ref_file:
            try:
                src = pathlib.Path(audio_ref_file)
                if src.exists():
                    ext = src.suffix.lower() or ".wav"
                    audio_ref_name = f"input_audio_{uuid.uuid4().hex[:10]}{ext}"
                    shutil.copy2(src, INPUT / audio_ref_name)
            except Exception as e:
                print(f"[audio_ref] failed to copy: {e}", flush=True)
                audio_ref_name = None

        if msr_original:
            workflow = _inject_runexx_params(
                _runexx_workflow_template(),
                ref1_image_name=image_name,
                ref2_image_name=msr_ref2_name,
                bg_image_name=msr_bg_name,
                prompt=prompt.strip(),
                negative_prompt=negative_prompt.strip() or DEFAULT_NEGATIVE,
                seed=seed_value,
                width=width,
                height=height,
                frames=frames,
                msr_frame_count=int(msr_frame_count),
            )
        else:
            workflow = _inject_params(
                _workflow_template(),
                preset=preset,
                image_name=image_name,
                prompt=prompt.strip(),
                negative_prompt=negative_prompt.strip() or DEFAULT_NEGATIVE,
                seed=seed_value,
                width=width,
                height=height,
                frames=frames,
                mode=mode,
                face_bbox=face_bbox,
                likeness_strength=likeness_strength,
                likeness_anchor_strength=likeness_anchor_strength,
                latent_anchor_strength=latent_anchor_strength,
                first_frame_strength=first_frame_strength,
                sulphur_lora_strength=sulphur_lora_strength,
                sulphur_v1_lora_strength=sulphur_v1_lora_strength,
                vbvr_lora_strength=vbvr_lora_strength,
                dreamly_lora_strength=dreamly_lora_strength,
                synth_lora_strength=synth_lora_strength,
                plora_lora_strength=plora_lora_strength,
                singularity_lora_strength=singularity_lora_strength,
                omninft_lora_strength=omninft_lora_strength,
                omninft_bf16_lora_strength=omninft_bf16_lora_strength,
                better_motion_lora_strength=better_motion_lora_strength,
                physics_v2_lora_strength=physics_v2_lora_strength,
                hardcut_lora_strength=hardcut_lora_strength,
                transition_lora_strength=transition_lora_strength,
                slot7_lora_strength=slot7_lora_strength,
                sulphur_audio_strength=sulphur_audio_strength,
                sulphur_v1_audio_strength=sulphur_v1_audio_strength,
                vbvr_audio_strength=vbvr_audio_strength,
                dreamly_audio_strength=dreamly_audio_strength,
                synth_audio_strength=synth_audio_strength,
                plora_audio_strength=plora_audio_strength,
                singularity_audio_strength=singularity_audio_strength,
                omninft_audio_strength=omninft_audio_strength,
                omninft_bf16_audio_strength=omninft_bf16_audio_strength,
                better_motion_audio_strength=better_motion_audio_strength,
                physics_v2_audio_strength=physics_v2_audio_strength,
                hardcut_audio_strength=hardcut_audio_strength,
                transition_audio_strength=transition_audio_strength,
                slot7_audio_strength=slot7_audio_strength,
                slot1_lora_file=slot1_lora_file,
                slot2_lora_file=slot2_lora_file,
                slot3_lora_file=slot3_lora_file,
                slot4_lora_file=slot4_lora_file,
                slot5_lora_file=slot5_lora_file,
                slot6_lora_file=slot6_lora_file,
                slot7_lora_file=slot7_lora_file,
                cache_at_step=int(cache_at_step),
                cache_warmup=int(cache_warmup),
                energy_threshold=float(energy_threshold),
                anchor_similarity_threshold=float(anchor_similarity_threshold),
                sigma_string=str(sigma_string or _SIGMA_TUNED),
                msr_enabled=msr_enabled,
                msr_ref2_name=msr_ref2_name,
                msr_ref3_name=msr_ref3_name,
                msr_ref4_name=msr_ref4_name,
                msr_bg_name=msr_bg_name,
                msr_frame_count=int(msr_frame_count),
                msr_guide_strength=float(msr_guide_strength),
                msr_lora_strength=float(msr_lora_strength),
                prompt_relay_enabled=bool(prompt_relay_enabled),
                prompt_segments=str(prompt_segments or ""),
                scene_chain_enabled=bool(scene_chain_enabled),
                scene_chain_prompt=str(scene_chain_prompt or ""),
                scene_chain_max_scenes=int(scene_chain_max_scenes),
                scene_chain_frame_overlap=int(scene_chain_frame_overlap),
                scene_chain_mid_guide=bool(scene_chain_mid_guide),
                scene_chain_mid_guide_strength=float(scene_chain_mid_guide_strength),
                kv_enabled=bool(kv_enabled),
                kv_strength=float(kv_strength),
                audio_ref_enabled=bool(audio_ref_enabled),
                audio_ref_filename=audio_ref_name,
                audio_ref_guidance_scale=float(audio_ref_guidance_scale),
                audio_ref_stem_sep=bool(audio_ref_stem_sep),
                audio_ref_normalize=bool(audio_ref_normalize),
            )

        mode_label = " (MSR-original)" if msr_original else (" (MSR)" if msr_enabled else "")
        progress(0.15, desc=f"generating {width}x{height}, {frames} frames + audio{mode_label}")
        print(
            f"[gen] {width}x{height} {frames}f seed={seed_value} mode={mode} "
            f"preset={preset} sigmas={repr(sigma_string[:20])} face={mode} "
            f"kv={kv_enabled}@{kv_strength:.2f} "
            f"sulphur_fro99={sulphur_lora_strength:.2f}/{sulphur_audio_strength:.2f} "
            f"sulphur_v1={sulphur_v1_lora_strength:.2f}/{sulphur_v1_audio_strength:.2f} "
            f"vbvr={vbvr_lora_strength:.2f}/{vbvr_audio_strength:.2f} "
            f"dreamly={dreamly_lora_strength:.2f}/{dreamly_audio_strength:.2f} "
            f"synth={synth_lora_strength:.2f}/{synth_audio_strength:.2f} "
            f"plora={plora_lora_strength:.2f}/{plora_audio_strength:.2f} "
            f"singularity={singularity_lora_strength:.2f}/{singularity_audio_strength:.2f} "
            f"omninft={omninft_lora_strength:.2f}/{omninft_audio_strength:.2f} "
            f"omninft_bf16={omninft_bf16_lora_strength:.2f}/{omninft_bf16_audio_strength:.2f} "
            f"better_motion={better_motion_lora_strength:.2f}/{better_motion_audio_strength:.2f} "
            f"physics_v2={physics_v2_lora_strength:.2f}/{physics_v2_audio_strength:.2f} "
            f"hardcut={hardcut_lora_strength:.2f}/{hardcut_audio_strength:.2f} "
            f"transition={transition_lora_strength:.2f}/{transition_audio_strength:.2f} "
            f"slot7={slot7_lora_strength:.2f}/{slot7_audio_strength:.2f} "
            f"likeness={likeness_strength:.2f} "
            f"like_anchor={likeness_anchor_strength:.2f} "
            f"lat_anchor={latent_anchor_strength:.2f} "
            f"first_frame={first_frame_strength:.2f} "
            f"anchor_sim={anchor_similarity_threshold:.2f} "
            f"energy={energy_threshold:.2f} "
            f"cache_step={cache_at_step} cache_warm={cache_warmup} "
            f"relay={prompt_relay_enabled} input_mode={input_mode!r} "
            f"scene_chain={scene_chain_enabled} max={scene_chain_max_scenes} "
            f"overlap={scene_chain_frame_overlap} mid={scene_chain_mid_guide}@{scene_chain_mid_guide_strength:.2f} "
            f"audio_ref={audio_ref_enabled}@{audio_ref_guidance_scale:.1f} "
            f"audio_stem_sep={audio_ref_stem_sep} "
            f"audio_norm={audio_ref_normalize} "
            f"audio_file={bool(audio_ref_name)}",
            flush=True,
        )
        result = _execute_workflow(workflow)

        out_dir = pathlib.Path(tempfile.mkdtemp())
        out_path = out_dir / "output.mp4"
        rc = subprocess.run(
            [
                _ffmpeg_exe(),
                "-y",
                "-i",
                result,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "24",
                str(out_path),
            ],
            capture_output=True,
            timeout=180,
        )
        final = str(out_path if rc.returncode == 0 and out_path.exists() else result)
        return final, f"{width}x{height}, {frames} frames, seed {seed_value}", seed_value
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        return None, tb[-6000:], seed_value


if os.environ.get("SKIP_STARTUP_SETUP") != "1":
    _ensure_comfy()
    _ensure_models()
    # Pre-download enhancer weights at startup so the download never happens
    # inside an @spaces.GPU fork (which would burn zerogpu quota on pure
    # network transfer). Disk-only ops here, no GPU needed.
    _ensure_enhancer()
    # Pre-populate workflow caches in the parent process so every @spaces.GPU
    # fork inherits the already-converted dicts via copy-on-write instead of
    # re-parsing + re-converting on every generation. Requires comfy nodes
    # initialized first (the converters look up NODE_CLASS_MAPPINGS for
    # widget param schemas).
    try:
        _init_comfy_nodes()
        _workflow_template()
        _runexx_workflow_template()
    except Exception as exc:
        print(f"[startup] workflow cache pre-populate failed: {exc}", flush=True)


def apply_preset(preset: str):
    p = PRESET_VALUES.get(preset, PRESET_VALUES["tuned"])
    return (
        gr.update(value=p["mode"]),
        gr.update(value=p["sulphur_fro99"]),
        gr.update(value=p["sulphur_v1"]),
        gr.update(value=p["vbvr"]),
        gr.update(value=p["dreamly"]),
        gr.update(value=p["synth"]),
        gr.update(value=p["plora"]),
        gr.update(value=p["singularity"]),
        gr.update(value=p["omninft"]),
        gr.update(value=p["omninft_bf16"]),
        gr.update(value=p["better_motion"]),
        gr.update(value=p["physics_v2"]),
        gr.update(value=p["hardcut"]),
        gr.update(value=p["transition"]),
        gr.update(value=p["likeness_strength"]),
        gr.update(value=p["likeness_anchor_strength"]),
        gr.update(value=p["latent_anchor_strength"]),
        gr.update(value=p["first_frame_strength"]),
        gr.update(value=p["anchor_similarity_threshold"]),
        gr.update(value=p["energy_threshold"]),
        gr.update(value=p["cache_warmup"]),
        gr.update(value=p["sigma_string"]),
    )


with gr.Blocks(title="10Eros LTX 2.3 image-to-video") as demo:
    gr.Markdown(
        "# 10Eros LTX 2.3 image-to-video\n"
        "huggingface space using comfyui backend for 10eros LTX 2.3 fp8 mixed "
        "checkpoint for I2V with native audio. make sure to upload a starting image "
        "first, write a prompt, optionally try a different preset, press enhance prompt to "
        "expand a short concept into a detailed video prompt tuned specifically for "
        "LTX. native audio is generated jointly with video. If your generations "
        "get limited by ZeroGPU duration, feel free to modify the ZeroGPU budget section.\n"
        "*you are solely responsible for all content you generate.*",
        line_breaks=True,
    )
    INPUT_MODE_I2V = "single image (i2v)"
    INPUT_MODE_MSR = "multi-reference (MSR)"
    INPUT_MODE_MSR_ORIGINAL = "multi-reference (original)"
    with gr.Row():
        with gr.Column():
            # input_mode + msr_* components retained as hidden so the proxy
            # payload positions stay stable and the underlying MSR injection
            # logic can be re-enabled in future without restructuring the
            # workflow. Permanently defaulted to single-image i2v.
            input_mode = gr.Radio(
                [
                    (INPUT_MODE_I2V, INPUT_MODE_I2V),
                    (f"{INPUT_MODE_MSR} (WIP)", INPUT_MODE_MSR),
                    (f"{INPUT_MODE_MSR_ORIGINAL} (WIP)", INPUT_MODE_MSR_ORIGINAL),
                ],
                value=INPUT_MODE_I2V,
                visible=False,
                label="input mode",
            )
            image = gr.Image(label="reference image", type="filepath")
            # MSR-only image slots: kept as hidden components so the workflow
            # injection chain still has placeholders if MSR is re-enabled.
            msr_ref2 = gr.Image(label="reference 2 (MSR)", type="filepath", visible=False)
            msr_ref3 = gr.Image(label="reference 3 (MSR)", type="filepath", visible=False)
            msr_ref4 = gr.Image(label="reference 4 (MSR)", type="filepath", visible=False)
            msr_background = gr.Image(label="background (MSR)", type="filepath", visible=False)
            prompt = gr.Textbox(label="prompt", lines=4)
            enhance_btn = gr.Button(
                "enhance prompt",
                variant="secondary",
                size="sm",
            )
            preset = gr.Dropdown(PRESETS, value="tuned", label="preset (sets all lora, targeting, and sigma defaults)")
            prompt_relay_enabled = gr.Checkbox(
                value=False,
                label="enable prompt relay (timeline-based prompts)",
            )
            prompt_segments = gr.Textbox(
                visible=False,
                lines=4,
                label="prompt segments",
                placeholder=(
                    "0-2: wide shot of city skyline at dusk\n"
                    "2-5: camera zooms into apartment window\n"
                    "5-8: a man at a desk turns to face the camera"
                ),
            )
            prompt_relay_help = gr.Markdown(
                visible=False,
                value=(
                    "**how to use:** `start-end: prompt text` lines are "
                    "accepted and converted to the official smart node syntax. "
                    "you can also use native prompt relay syntax like "
                    "`prompt one [0-50] | prompt two [50-100]` or `Scene 1:` "
                    "blocks. the main prompt above acts as the global anchor "
                    "across the whole video. prompt relay is disabled in any "
                    "multi-reference mode."
                ),
            )
            negative = gr.Textbox(label="negative prompt", value=DEFAULT_NEGATIVE, lines=2)
            seconds = gr.Slider(1.0, 41.0, value=4.0, step=0.5, label="duration (seconds, up to ~1000 frames)")
            with gr.Accordion("loras", open=False):
                sulphur_lora_strength = gr.Slider(
                    0.0, 1.0, value=0.15, step=0.05,
                    label="sulphur fro99 (small + fast, 0 = off)",
                )
                sulphur_v1_lora_strength = gr.Slider(
                    0.0, 1.0, value=0.15, step=0.05,
                    label="sulphur v1 (full precision newest, 0 = off)",
                )
                vbvr_lora_strength = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05,
                    label="vbvr lora (0 = off, 0.5 works good)",
                )
                dreamly_lora_strength = gr.Slider(
                    0.0, 1.0, value=0.6, step=0.05,
                    label="dreamly lora (0 = off)",
                )
                _slot2_orig = _slot_original("slot2", SYNTH_LORA_FILENAME)
                with gr.Row(visible=bool(_slot2_orig)):
                    synth_lora_strength = gr.Slider(
                        0.0, 1.0, value=0.0, step=0.05,
                        label="synth lora (0 = off)",
                    )
                slot2_lora_file = gr.State(value=_slot2_orig or LORA_NONE)
                _slot3_orig = _slot_original("slot3", PLORA_LORA_FILENAME)
                with gr.Row(visible=bool(_slot3_orig)):
                    plora_lora_strength = gr.Slider(
                        0.0, 1.0, value=0.0, step=0.05,
                        label="plora (0 = off)",
                    )
                slot3_lora_file = gr.State(value=_slot3_orig or LORA_NONE)
                singularity_lora_strength = gr.Slider(
                    0.0, 1.0, value=0.3, step=0.05,
                    label="singularity (0 = off)",
                )
                omninft_lora_strength = gr.Slider(
                    0.0, 2.0, value=0.8, step=0.05,
                    label="omninft converted (0 = off, default 0.8)",
                )
                _slot4_orig = _slot_original("slot4", OMNINFT_BF16_LORA_FILENAME)
                with gr.Row(visible=bool(_slot4_orig)):
                    omninft_bf16_lora_strength = gr.Slider(
                        0.0, 2.0, value=0.0, step=0.05,
                        label="omninft RL bf16 / kijai (0 = off)",
                    )
                slot4_lora_file = gr.State(value=_slot4_orig or LORA_NONE)
                _slot5_orig = _slot_original("slot5", BETTER_MOTION_LORA_FILENAME)
                with gr.Row(visible=bool(_slot5_orig)):
                    better_motion_lora_strength = gr.Slider(
                        0.0, 1.0, value=0.0, step=0.05,
                        label="better motion / mistic (0 = off)",
                    )
                slot5_lora_file = gr.State(value=_slot5_orig or LORA_NONE)
                _slot6_orig = _slot_original("slot6", PHYSICS_V2_LORA_FILENAME)
                with gr.Row(visible=bool(_slot6_orig)):
                    physics_v2_lora_strength = gr.Slider(
                        0.0, 1.0, value=0.0, step=0.05,
                        label="physics v2 / mistic (0 = off)",
                    )
                slot6_lora_file = gr.State(value=_slot6_orig or LORA_NONE)
                _slot1_orig = _slot_original("slot1", HARDCUT_LORA_FILENAME)
                with gr.Row(visible=bool(_slot1_orig)):
                    hardcut_lora_strength = gr.Slider(
                        0.0, 1.0, value=0.0, step=0.05,
                        label="cinematic hardcut (0 = off)",
                    )
                slot1_lora_file = gr.State(value=_slot1_orig or LORA_NONE)
                transition_lora_strength = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.05,
                    label="transition lora (0 = off, default 0.15)",
                )
                # Slot 7 oculto — mantenido solo para estabilidad de la firma
                with gr.Row(visible=False):
                    slot7_lora_strength = gr.Slider(
                        0.0, 1.0, value=0.0, step=0.05, label="slot 7 (hidden)"
                    )
                slot7_lora_file = gr.State(value=LORA_NONE)
                # ── Custom LoRAs: una fila por archivo instalado (hasta 6) ──
                _MAX_CUSTOM_SLOTS = 6
                _custom_dir = MODELS / "loras" / "ltx23" / "custom"
                _custom_manifest: dict = {}
                if (_cmf := _custom_dir / "manifest.json").exists():
                    try:
                        _custom_manifest = json.loads(_cmf.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                _custom_files: list[tuple[str, str]] = []
                if _custom_dir.exists():
                    for _cf in sorted(_custom_dir.glob("*.safetensors")):
                        _cname = (_custom_manifest.get(_cf.name) or {}).get("name") or _cf.stem
                        _custom_files.append((_cname, f"ltx23/custom/{_cf.name}"))
                if _custom_files:
                    gr.Markdown("**custom LoRAs** (instala más en la Celda 5)")
                _custom_vid_sliders: list = []
                _custom_aud_sliders: list = []
                _custom_file_states: list = []
                for _ci in range(_MAX_CUSTOM_SLOTS):
                    if _ci < len(_custom_files):
                        _cname, _cfile = _custom_files[_ci]
                        with gr.Row():
                            _cvs = gr.Slider(
                                0.0, 1.0, value=0.7, step=0.05,
                                label=f"{_cname} — video (0=off)",
                            )
                            _cas = gr.Slider(
                                0.0, 1.0, value=0.7, step=0.05,
                                label=f"{_cname} — audio (0=off)",
                            )
                        _cfs = gr.State(value=_cfile)
                    else:
                        with gr.Row(visible=False):
                            _cvs = gr.Slider(
                                0.0, 1.0, value=0.0, step=0.05,
                                label=f"custom {_ci+1} (unused)",
                            )
                            _cas = gr.Slider(
                                0.0, 1.0, value=0.0, step=0.05,
                                label=f"custom {_ci+1} audio (unused)",
                            )
                        _cfs = gr.State(value=LORA_NONE)
                    _custom_vid_sliders.append(_cvs)
                    _custom_aud_sliders.append(_cas)
                    _custom_file_states.append(_cfs)
            with gr.Accordion("resolution", open=False):
                with gr.Row():
                    target_mp = gr.Number(
                        value=1.15, minimum=0.1, maximum=4.0, precision=4,
                        label="target megapixels",
                    )
                    snap_multiple = gr.Radio(
                        [("32", 32), ("64 (recommended)", 64)], value=64,
                        label="snap to multiple",
                    )
                custom_res_enabled = gr.Checkbox(
                    value=False,
                    label="custom resolution (overrides megapixels)",
                )
                with gr.Row(visible=False) as custom_res_row:
                    max_width = gr.Slider(512, 1536, value=1120, step=64, label="max width")
                    max_height = gr.Slider(512, 1536, value=1344, step=64, label="max height")
            with gr.Accordion("targeting", open=False):
                mode = gr.Radio(["anchor only", "auto face", "manual bbox"], value="anchor only", label="face mode")
                face_bbox = gr.Textbox(label="manual bbox", placeholder="x1,y1,x2,y2, normalized 0-1")
                likeness_strength = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="likeness guide")
                likeness_anchor_strength = gr.Slider(0.0, 1.0, value=0.15, step=0.01, label="likeness anchor")
                latent_anchor_strength = gr.Slider(0.0, 0.5, value=0.08, step=0.01, label="latent anchor")
                first_frame_strength = gr.Slider(0.0, 1.0, value=0.82, step=0.01, label="first frame strength")
            with gr.Accordion("funpack", open=False):
                kv_enabled = gr.Checkbox(
                    value=False,
                    label="enable K/V identity conditioning (experimental)",
                )
                kv_strength = gr.Slider(
                    0.0, 2.0, value=1.0, step=0.05,
                    label="K/V strength (0 = off, 1 = funpack default, >1 = stronger identity)",
                )
                with gr.Accordion("scene chaining (experimental)", open=False):
                    scene_chain_enabled = gr.Checkbox(
                        value=False,
                        label="enable scene chaining (bypasses pass 2 for v1)",
                    )
                    scene_chain_prompt = gr.Textbox(
                        lines=7,
                        label="scene chain prompt",
                        placeholder=(
                            "Scene 1:\n"
                            "same person from the reference image, close-up, clear facial detail\n\n"
                            "Scene 2:\n"
                            "same person walking through a neon alley, rain reflections, face remains recognizable"
                        ),
                    )
                    scene_chain_max_scenes = gr.Slider(
                        2, 4, value=2, step=1,
                        label="max scene chunks (free-tier test: keep at 2)",
                    )
                    scene_chain_frame_overlap = gr.Slider(
                        0, 24, value=8, step=8,
                        label="scene overlap frames (8 = safer first test)",
                    )
                    scene_chain_mid_guide = gr.Checkbox(
                        value=True,
                        label="carry previous-scene midpoint as guide",
                    )
                    scene_chain_mid_guide_strength = gr.Slider(
                        0.25, 0.5, value=0.25, step=0.05,
                        label="mid-scene guide strength",
                    )
            with gr.Accordion("audio", open=False):
                audio_ref_enabled = gr.Checkbox(
                    value=False,
                    label="audio reference (voice ID transfer)",
                )
                audio_ref_guidance_scale = gr.Slider(
                    0.0, 10.0, value=3.0, step=0.1,
                    label="identity guidance scale (lower if audio problems)",
                )
                audio_ref_stem_sep = gr.Checkbox(
                    value=False,
                    label="isolate voice from background (stem separation, slower)",
                )
                audio_ref_normalize = gr.Checkbox(
                    value=True,
                    label="normalize reference audio (caps to 10s, boosts quiet clips)",
                )
                audio_ref_file = gr.Audio(
                    type="filepath",
                    label="audio reference (~4s clip recommended)",
                )
                with gr.Accordion("per-lora audio strength (advanced)", open=False):
                    gr.Markdown(
                        "controls how each lora affects the **audio** stream "
                        "(loras default to applying equally to video + audio). "
                        "set to 0 to stop a lora from influencing audio while "
                        "keeping its video effect."
                    )
                    sulphur_audio_strength = gr.Slider(
                        0.0, 1.0, value=0.15, step=0.05,
                        label="sulphur fro99 (audio)",
                    )
                    sulphur_v1_audio_strength = gr.Slider(
                        0.0, 1.0, value=0.15, step=0.05,
                        label="sulphur v1 (audio)",
                    )
                    vbvr_audio_strength = gr.Slider(
                        0.0, 1.0, value=0.5, step=0.05,
                        label="vbvr (audio)",
                    )
                    dreamly_audio_strength = gr.Slider(
                        0.0, 1.0, value=0.6, step=0.05,
                        label="dreamly (audio)",
                    )
                    with gr.Row(visible=bool(_slot2_orig)):
                        synth_audio_strength = gr.Slider(
                            0.0, 1.0, value=0.0, step=0.05,
                            label="synth (audio)",
                        )
                    with gr.Row(visible=bool(_slot3_orig)):
                        plora_audio_strength = gr.Slider(
                            0.0, 1.0, value=0.0, step=0.05,
                            label="plora (audio)",
                        )
                    singularity_audio_strength = gr.Slider(
                        0.0, 1.0, value=0.3, step=0.05,
                        label="singularity (audio)",
                    )
                    omninft_audio_strength = gr.Slider(
                        0.0, 2.0, value=0.8, step=0.05,
                        label="omninft converted (audio)",
                    )
                    with gr.Row(visible=bool(_slot4_orig)):
                        omninft_bf16_audio_strength = gr.Slider(
                            0.0, 2.0, value=0.0, step=0.05,
                            label="omninft RL bf16 / kijai (audio)",
                        )
                    with gr.Row(visible=bool(_slot5_orig)):
                        better_motion_audio_strength = gr.Slider(
                            0.0, 1.0, value=0.0, step=0.05,
                            label="better motion / mistic (audio)",
                        )
                    with gr.Row(visible=bool(_slot6_orig)):
                        physics_v2_audio_strength = gr.Slider(
                            0.0, 1.0, value=0.0, step=0.05,
                            label="physics v2 / mistic (audio)",
                        )
                    with gr.Row(visible=bool(_slot1_orig)):
                        hardcut_audio_strength = gr.Slider(
                            0.0, 1.0, value=0.0, step=0.05,
                            label="cinematic hardcut (audio)",
                        )
                    transition_audio_strength = gr.Slider(
                        0.0, 1.0, value=0.0, step=0.05,
                        label="transition lora (audio)",
                    )
                    with gr.Row(visible=False):
                        slot7_audio_strength = gr.Slider(
                            0.0, 1.0, value=0.0, step=0.05,
                            label="slot 7 (audio, hidden)",
                        )
            with gr.Accordion("multi-reference settings (MSR)", open=False, visible=False) as msr_settings_acc:
                msr_frame_count = gr.Dropdown(
                    [17, 25, 33, 41], value=41,
                    label="pseudo-video frame count (41 = max identity reinforcement; lower = faster)",
                )
                msr_guide_strength = gr.Slider(
                    0.0, 1.0, value=1.0, step=0.05,
                    label="MSR guide strength (LTXAddVideoICLoRAGuide)",
                )
                msr_lora_strength = gr.Slider(
                    0.0, 1.0, value=0.7, step=0.05,
                    label="MSR ic-lora strength (0.5-1.0 safe band)",
                )
            with gr.Accordion("identity tuning (advanced)", open=False):
                anchor_similarity_threshold = gr.Slider(
                    0.0, 1.0, value=0.3, step=0.05,
                    label="similarity threshold (lower = corrects drift earlier, catches face changes on angles; too low can distort anatomy)",
                )
                cache_at_step = gr.Slider(
                    0, 12, value=0, step=1,
                    label="anchor cache step (0 = auto-align to frame count; controls when identity locks)",
                )
                cache_warmup = gr.Slider(
                    10, 2000, value=400, step=10,
                    label="cache warmup (affects sustained identity over duration; 50/400/1000 behave differently)",
                )
                energy_threshold = gr.Slider(
                    0.0, 1.0, value=0.3, step=0.05,
                    label="energy threshold (latent anchor sensitivity)",
                )
                sigma_string = gr.Textbox(
                    value=_SIGMA_TUNED,
                    placeholder="comma-separated decreasing values in [0,1] ending at 0",
                    label="refine sigmas",
                )
            with gr.Accordion("zerogpu budget", open=False):
                enhance_budget = gr.Slider(
                    20, 540, value=DEFAULT_ENHANCE_BUDGET, step=10,
                    label="enhance prompt budget (seconds)",
                )
                gen_budget = gr.Slider(
                    0, 540, value=0, step=10,
                    label="generation budget (seconds, 0 = automatic)",
                )
            with gr.Row():
                seed = gr.Number(label="seed", value=-1, precision=0)
                randomize = gr.Checkbox(label="randomize seed", value=True)
            button = gr.Button("generate", variant="primary")
        with gr.Column():
            video = gr.Video(label="output")
            status = gr.Textbox(label="status", interactive=False)
            used_seed = gr.Number(label="used seed", interactive=False)

    button.click(
        fn=generate,
        inputs=[
            image,
            prompt,
            negative,
            preset,
            seconds,
            max_width,
            max_height,
            mode,
            face_bbox,
            likeness_strength,
            likeness_anchor_strength,
            latent_anchor_strength,
            first_frame_strength,
            seed,
            randomize,
            gen_budget,
            target_mp,
            snap_multiple,
            custom_res_enabled,
            sulphur_lora_strength,
            sulphur_v1_lora_strength,
            vbvr_lora_strength,
            dreamly_lora_strength,
            synth_lora_strength,
            plora_lora_strength,
            singularity_lora_strength,
            omninft_lora_strength,
            omninft_bf16_lora_strength,
            better_motion_lora_strength,
            physics_v2_lora_strength,
            hardcut_lora_strength,
            transition_lora_strength,
            slot7_lora_strength,
            sulphur_audio_strength,
            sulphur_v1_audio_strength,
            vbvr_audio_strength,
            dreamly_audio_strength,
            synth_audio_strength,
            plora_audio_strength,
            singularity_audio_strength,
            omninft_audio_strength,
            omninft_bf16_audio_strength,
            better_motion_audio_strength,
            physics_v2_audio_strength,
            hardcut_audio_strength,
            transition_audio_strength,
            slot7_audio_strength,
            slot1_lora_file,
            slot2_lora_file,
            slot3_lora_file,
            slot4_lora_file,
            slot5_lora_file,
            slot6_lora_file,
            slot7_lora_file,
            cache_at_step,
            cache_warmup,
            energy_threshold,
            anchor_similarity_threshold,
            sigma_string,
            input_mode,
            msr_ref2,
            msr_ref3,
            msr_ref4,
            msr_background,
            msr_frame_count,
            msr_guide_strength,
            msr_lora_strength,
            prompt_relay_enabled,
            prompt_segments,
            scene_chain_enabled,
            scene_chain_prompt,
            scene_chain_max_scenes,
            scene_chain_frame_overlap,
            scene_chain_mid_guide,
            scene_chain_mid_guide_strength,
            kv_enabled,
            kv_strength,
            audio_ref_enabled,
            audio_ref_file,
            audio_ref_guidance_scale,
            audio_ref_stem_sep,
            audio_ref_normalize,
            *_custom_vid_sliders,
            *_custom_aud_sliders,
            *_custom_file_states,
        ],
        outputs=[video, status, used_seed],
    )

    enhance_btn.click(
        fn=enhance_prompt,
        inputs=[image, prompt, enhance_budget,
                msr_ref2, msr_ref3, msr_ref4, msr_background],
        outputs=[prompt],
    )

    preset.change(
        fn=apply_preset,
        inputs=[preset],
        outputs=[
            mode,
            sulphur_lora_strength, sulphur_v1_lora_strength, vbvr_lora_strength,
            dreamly_lora_strength, synth_lora_strength, plora_lora_strength,
            singularity_lora_strength, omninft_lora_strength, omninft_bf16_lora_strength,
            better_motion_lora_strength, physics_v2_lora_strength, hardcut_lora_strength,
            transition_lora_strength,
            likeness_strength, likeness_anchor_strength, latent_anchor_strength,
            first_frame_strength, anchor_similarity_threshold, energy_threshold,
            cache_warmup, sigma_string,
        ],
    )

    def _on_input_mode_change(m: str):
        # MSR modes reveal extra image slots + MSR settings accordion + relabel
        # the main image as "reference 1". The original-workflow mode supports
        # only ref1 + ref2 + background (LiconMSR slots actually wired by that
        # workflow), so ref3/ref4 stay hidden in that mode.
        # Registered LAST so /generate and /enhance_prompt fn_indexes remain
        # stable for the proxy client.
        is_msr_ours = m == "multi-reference (MSR)"
        is_msr_original = m == "multi-reference (original)"
        any_msr = is_msr_ours or is_msr_original
        return (
            gr.update(label="reference 1" if any_msr else "reference image"),
            gr.update(visible=any_msr),
            gr.update(visible=is_msr_ours),
            gr.update(visible=is_msr_ours),
            gr.update(visible=any_msr),
            gr.update(visible=any_msr),
        )

    input_mode.change(
        fn=_on_input_mode_change,
        inputs=[input_mode],
        outputs=[image, msr_ref2, msr_ref3, msr_ref4, msr_background,
                 msr_settings_acc],
    )

    def _on_prompt_relay_toggle(enabled: bool):
        # Registered LAST so it takes the highest fn_index and doesn't shift
        # /generate, /enhance_prompt, or any other handler the proxy depends
        # on. Toggles visibility of the segments textbox + helper markdown.
        return (
            gr.update(visible=bool(enabled)),
            gr.update(visible=bool(enabled)),
        )

    prompt_relay_enabled.change(
        fn=_on_prompt_relay_toggle,
        inputs=[prompt_relay_enabled],
        outputs=[prompt_segments, prompt_relay_help],
    )

    custom_res_enabled.change(
        fn=lambda enabled: gr.update(visible=bool(enabled)),
        inputs=[custom_res_enabled],
        outputs=[custom_res_row],
    )

demo.queue()

if __name__ == "__main__":
    demo.launch()
