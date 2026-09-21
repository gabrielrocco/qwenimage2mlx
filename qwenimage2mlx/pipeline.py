"""QwenImage21 — text-to-image pipeline in MLX with optional q8/q4 quant."""
import os
import time

import numpy as np
import mlx.core as mx

from .config import ORIGINAL_DIR, SYS_PROMPT, TEMPLATE_T2I, quant_dir
from .download import ensure_original
from . import quantize as Q
from .weights import build_transformer, build_vae, build_text_encoder


class QwenImage21:
    def __init__(self, quantize=None, dtype=mx.bfloat16, model_dir=None, save=True,
                 verbose=True):
        """
        quantize: None (fp16), 'q8' or 'q4'.
        save:     when quantizing, save the quant to cache and reuse it next time.
        model_dir: override the original weights dir (defaults to HF cache download).
        """
        self.dtype = dtype
        self.verbose = verbose
        self._log("Resolving weights...")
        model_dir = model_dir or ensure_original()
        self.model_dir = model_dir

        from transformers import AutoTokenizer
        from diffusers import FlowMatchEulerDiscreteScheduler

        self.tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "processor"))
        sys_block = f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n"
        self.drop_idx = len(self.tok(sys_block)["input_ids"])
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            os.path.join(model_dir, "scheduler"))

        bits = {"q8": 8, "q4": 4, None: None}[quantize]

        self._log("Loading VAE (fp16)...")
        self.vae, vcfg = build_vae(model_dir, dtype)
        self.latents_mean = mx.array(vcfg["latents_mean"], dtype=mx.float32)
        self.latents_std = mx.array(vcfg["latents_std"], dtype=mx.float32)

        if bits is None:
            self._log("Loading transformer + text encoder (fp16)...")
            self.transformer, self.tcfg = build_transformer(model_dir, dtype)
            self.text_encoder, _ = build_text_encoder(model_dir, dtype)
        else:
            dest = quant_dir(bits)
            if Q.has_saved_quant(dest):
                self._log(f"Loading cached {quantize} from {dest} ...")
                self.transformer, self.tcfg = build_transformer(model_dir, dtype)
                self.text_encoder, _ = build_text_encoder(model_dir, dtype)
                Q.load_quant_into(self.transformer, self.text_encoder, dest)
            else:
                self._log(f"Quantizing to {quantize} (first run)...")
                self.transformer, self.tcfg = build_transformer(model_dir, dtype)
                self.text_encoder, _ = build_text_encoder(model_dir, dtype)
                Q.quantize_module(self.transformer, bits)
                Q.quantize_module(self.text_encoder, bits)
                if save:
                    self._log(f"Saving {quantize} to {dest} ...")
                    Q.save_quant(bits, self.transformer, self.text_encoder, dest)
        self._log("Ready.")

    def _log(self, m):
        if self.verbose:
            print(f"[qwenimage2mlx] {m}", flush=True)

    def _encode_prompt(self, prompt):
        text = TEMPLATE_T2I.format(prompt if prompt else " ")
        ids = self.tok(text, return_tensors="np")["input_ids"].astype(np.int32)
        _, hs = self.text_encoder(mx.array(ids), None, output_hidden_states=True)
        hidden = hs[-1][:, self.drop_idx:, :]  # last layer pre-final-norm, prefix stripped
        return hidden.astype(self.dtype)

    def _calc_shift(self, seq_len):
        c = self.scheduler.config
        b_seq, m_seq = c.get("base_image_seq_len", 256), c.get("max_image_seq_len", 4096)
        b_sh, m_sh = c.get("base_shift", 0.5), c.get("max_shift", 1.15)
        m = (m_sh - b_sh) / (m_seq - b_seq)
        return seq_len * m + (b_sh - m * b_seq)

    def generate(self, prompt, negative_prompt=" ", width=1024, height=1024,
                 steps=20, true_cfg_scale=4.0, seed=42, output=None,
                 open=False, save=True):
        """Generate an image. Returns a PIL.Image.

        output: path to save PNG (default ~/qwen21_out.png when save=True).
        open:   open the saved image after generation.
        save:   write the PNG to disk.
        """
        import torch
        lat_h, lat_w = 2 * (height // 32), 2 * (width // 32)
        num_ch, seq = 64, lat_h * lat_w

        pos = self._encode_prompt(prompt)
        do_cfg = true_cfg_scale > 1
        neg = self._encode_prompt(negative_prompt) if do_cfg else None
        img_shapes = [(1, lat_h, lat_w)]

        def img_mask(txt_len):
            m = np.zeros((1, txt_len + seq // 4), dtype=bool)
            m[:, txt_len:] = True
            return mx.array(m)

        mx.random.seed(seed)
        latents = mx.random.normal((1, num_ch, lat_h, lat_w)).astype(mx.float32)
        latents = latents.reshape(1, num_ch, seq).transpose(0, 2, 1)

        sigmas = np.linspace(1.0, 1.0 / steps, steps)
        self.scheduler.set_timesteps(sigmas=sigmas, mu=self._calc_shift(seq), device="cpu")
        timesteps = self.scheduler.timesteps

        pmask, nmask = img_mask(pos.shape[1]), (img_mask(neg.shape[1]) if do_cfg else None)
        t0 = time.time()
        for i, t in enumerate(timesteps):
            ts = mx.array([float(t) / 1000.0], dtype=mx.float32)
            pred = self.transformer(latents.astype(self.dtype), pos, ts,
                                    img_shapes, pmask, None)[:, -seq:, :].astype(mx.float32)
            if do_cfg:
                npred = self.transformer(latents.astype(self.dtype), neg, ts,
                                         img_shapes, nmask, None)[:, -seq:, :].astype(mx.float32)
                pred = npred + true_cfg_scale * (pred - npred)
            out = self.scheduler.step(
                torch.tensor(np.array(pred)), t,
                torch.tensor(np.array(latents.astype(mx.float32))), return_dict=False)[0]
            latents = mx.array(out.numpy())
            mx.eval(latents)
            self._log(f"step {i+1}/{steps}")
        self._log(f"generated in {time.time()-t0:.1f}s")

        lat = latents.transpose(0, 2, 1).reshape(1, num_ch, lat_h, lat_w).astype(mx.float32)
        lat = lat * self.latents_std.reshape(1, num_ch, 1, 1) + self.latents_mean.reshape(1, num_ch, 1, 1)
        z = lat.transpose(0, 2, 3, 1)
        img = np.array(self.vae.decode(z.astype(self.dtype)).astype(mx.float32))
        img = img[0, :, :, :3]
        img = ((img + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)

        from PIL import Image
        pil = Image.fromarray(img)
        if save:
            output = output or os.path.expanduser("~/qwen21_out.png")
            pil.save(output)
            self._log(f"saved {output}")
            if open:
                os.system(f'open "{output}"')
        return pil
