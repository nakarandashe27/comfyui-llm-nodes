"""ComfyUI-обёртки над api.py. Здесь только конвертация тензоров и файлов."""
import io
import os
import time

import numpy as np
import torch
from PIL import Image

from . import api

_PROJECT_INPUT = ("STRING", {"default": "", "tooltip": "Имя проекта для учёта расходов (тег в дашборде). Можно оставить пустым."})


class LLMText:
    CATEGORY = "LLM"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "openrouter/google/gemini-2.5-flash"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "system": ("STRING", {"multiline": True, "default": ""}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 200000}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "project": _PROJECT_INPUT,
            },
        }

    def run(self, model, prompt, system="", max_tokens=1024, temperature=1.0, project=""):
        return (api.chat(model, prompt, system=system, max_tokens=max_tokens,
                         temperature=temperature, project=project),)


class LLMImage:
    CATEGORY = "LLM"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "nano-banana"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "size": ("STRING", {"default": "", "tooltip": "Напр. 1024x1024. Пусто — дефолт модели."}),
                "project": _PROJECT_INPUT,
            },
        }

    def run(self, model, prompt, size="", project=""):
        raw = api.image(model, prompt, size=size, project=project)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,]
        return (tensor,)


class LLMVideo:
    CATEGORY = "LLM"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "veo-3"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "seconds": ("INT", {"default": 8, "min": 1, "max": 60}),
                "size": ("STRING", {"default": "", "tooltip": "Напр. 1280x720. Пусто — дефолт модели."}),
                "input_reference": ("IMAGE",),
                "project": _PROJECT_INPUT,
            },
        }

    def run(self, model, prompt, seconds=8, size="", input_reference=None, project=""):
        ref_png = None
        if input_reference is not None:
            arr = (input_reference[0].cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, "PNG")
            ref_png = buf.getvalue()

        data = api.video(model, prompt, seconds=seconds, size=size,
                         input_reference_png=ref_png, project=project)

        import folder_paths  # модуль ComfyUI, доступен только внутри него
        path = os.path.join(folder_paths.get_output_directory(),
                            "llm_video_%d.mp4" % int(time.time()))
        with open(path, "wb") as f:
            f.write(data)
        return (path,)
