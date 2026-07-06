"""ComfyUI-обёртки над api.py. Здесь только конвертация тензоров и файлов."""
import io
import os
import time

import numpy as np
import torch
from PIL import Image

from . import api

_PROJECT_INPUT = ("STRING", {"default": "", "tooltip": "Имя проекта для учёта расходов (тег в дашборде). Можно оставить пустым."})


def _model_combo(mode, fallback):
    """Дропдаун моделей со шлюза; фолбэк-константы всегда в списке, чтобы
    сохранённые воркфлоу валидировались и при недоступном шлюзе."""
    return (sorted(set(api.list_models(mode)) | set(fallback)), {"default": fallback[0]})


def _tensor_to_png(image_tensor, index=0):
    arr = (image_tensor[index].cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


class LLMText:
    CATEGORY = "LLM"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": _model_combo("chat", ["gemini-2.5-flash", "claude-sonnet-5",
                                               "claude-sonnet-4.6", "gemini-2.5-flash-lite",
                                               "gpt-5.5"]),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "system": ("STRING", {"multiline": True, "default": ""}),
                "image": ("IMAGE", {"tooltip": "Опционально: включает vision — модель видит картинку (описание, анализ, промт по референсу)."}),
                "reasoning_effort": (["off", "minimal", "low", "medium", "high", "xhigh"],
                                     {"default": "off", "tooltip": "Глубина размышлений reasoning-моделей. off — параметр не отправляется."}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 200000}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1,
                                 "tooltip": "В API не уходит: смени (или randomize), чтобы перегенерировать с теми же входами."}),
                "project": _PROJECT_INPUT,
            },
        }

    def run(self, model, prompt, system="", image=None, reasoning_effort="off",
            max_tokens=1024, temperature=1.0, seed=0, project=""):
        # seed — только обход кэша ComfyUI (повторный запуск), в запрос не идёт
        png = _tensor_to_png(image) if image is not None else None
        return (api.chat(model, prompt, system=system, max_tokens=max_tokens,
                         temperature=temperature, project=project, image_png=png,
                         reasoning_effort=reasoning_effort),)


class LLMImage:
    CATEGORY = "LLM"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": _model_combo("image_generation", ["nano-banana", "nano-banana-pro"]),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "aspect_ratio": (["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "21:9"], {"default": "auto"}),
                "resolution": (["auto", "1K", "2K", "4K"], {"default": "auto"}),
                "image_1": ("IMAGE", {"tooltip": "Опционально: референс/исходник — включает режим редактирования (image-to-image)."}),
                "image_2": ("IMAGE", {"tooltip": "Опционально: второй референс (совмещение, перенос стиля)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1,
                                 "tooltip": "В API не уходит: смени (или randomize), чтобы перегенерировать с теми же входами."}),
                "project": _PROJECT_INPUT,
            },
        }

    def run(self, model, prompt, aspect_ratio="auto", resolution="auto",
            image_1=None, image_2=None, seed=0, project=""):
        # seed — только обход кэша ComfyUI, в запрос не идёт
        refs = [_tensor_to_png(t) for t in (image_1, image_2) if t is not None]
        raw = api.image(model, prompt, aspect_ratio=aspect_ratio, resolution=resolution,
                        input_images=refs or None, project=project)
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
                "model": _model_combo("video_generation", ["veo-3"]),
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
        ref_png = _tensor_to_png(input_reference) if input_reference is not None else None

        data = api.video(model, prompt, seconds=seconds, size=size,
                         input_reference_png=ref_png, project=project)

        import folder_paths  # модуль ComfyUI, доступен только внутри него
        path = os.path.join(folder_paths.get_output_directory(),
                            "llm_video_%d.mp4" % int(time.time()))
        with open(path, "wb") as f:
            f.write(data)
        return (path,)
