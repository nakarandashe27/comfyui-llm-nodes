"""ComfyUI-обёртки над api.py. Здесь только конвертация тензоров и файлов."""
import io
import os
import time

import numpy as np
import torch
from PIL import Image
from comfy_api.latest import io as comfy_io

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


def _text_model_options():
    """Опции DynamicCombo: у каждой модели — свой набор значений reasoning_effort.
    claude/gemini понимают low/medium/high, у gpt есть ещё minimal/xhigh;
    незнакомые модели получают полный список. off — параметр не отправляется."""
    models = sorted(set(api.list_models("chat")) | {"gemini-2.5-flash", "claude-sonnet-5",
                                                    "claude-sonnet-4.6", "gemini-2.5-flash-lite",
                                                    "gpt-5.5"})
    options = []
    for m in models:
        # ponytail: семейство по префиксу имени; метаданные шлюза (supports_reasoning)
        # ненадёжны — LiteLLM не знает новых openrouter-моделей
        if m.startswith("claude") or m.startswith("gemini"):
            efforts = ["off", "low", "medium", "high"]
        else:
            efforts = ["off", "minimal", "low", "medium", "high", "xhigh"]
        options.append(comfy_io.DynamicCombo.Option(m, [
            comfy_io.Combo.Input("reasoning_effort", options=efforts, default="off",
                                 tooltip="Глубина размышлений reasoning-моделей. off — параметр не отправляется."),
        ]))
    return options


class LLMText(comfy_io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
            node_id="LLMText",
            display_name="LLM Text",
            category="LLM",
            inputs=[
                comfy_io.DynamicCombo.Input("model", options=_text_model_options()),
                comfy_io.String.Input("prompt", multiline=True, default=""),
                comfy_io.String.Input("system", multiline=True, default="", optional=True),
                comfy_io.Image.Input("image", optional=True,
                                     tooltip="Опционально: включает vision — модель видит картинку (описание, анализ, промт по референсу)."),
                comfy_io.Int.Input("max_tokens", default=1024, min=1, max=200000, optional=True),
                comfy_io.Float.Input("temperature", default=1.0, min=0.0, max=2.0, step=0.05, optional=True),
                comfy_io.Int.Input("seed", default=0, min=0, max=2**31 - 1, optional=True,
                                   control_after_generate=True,
                                   tooltip="В API не уходит: смени (или randomize), чтобы перегенерировать с теми же входами."),
                comfy_io.String.Input("project", default="", optional=True,
                                      tooltip="Имя проекта для учёта расходов (тег в дашборде). Можно оставить пустым."),
            ],
            outputs=[comfy_io.String.Output()],
        )

    @classmethod
    def execute(cls, model, prompt, system="", image=None,
                max_tokens=1024, temperature=1.0, seed=0, project="") -> comfy_io.NodeOutput:
        # model — словарь DynamicCombo: сама модель + её reasoning_effort
        # seed — только обход кэша ComfyUI (повторный запуск), в запрос не идёт
        png = _tensor_to_png(image) if image is not None else None
        return comfy_io.NodeOutput(
            api.chat(model["model"], prompt, system=system, max_tokens=max_tokens,
                     temperature=temperature, project=project, image_png=png,
                     reasoning_effort=model.get("reasoning_effort", "off")))


def _image_model_options():
    """Опции DynamicCombo: у каждой модели — свой набор параметров.
    gpt-image-семейство — quality low/medium/high; остальные (gemini/nano-banana
    и незнакомые) — aspect_ratio + resolution. Новая модель со шлюза без правки
    нод получит дефолтный набор."""
    models = sorted(set(api.list_models("image_generation")) | {"nano-banana", "nano-banana-pro"})
    options = []
    for m in models:
        if "gpt" in m:  # ponytail: семейство по подстроке имени; таблица маппинга — когда эвристика соврёт
            params = [comfy_io.Combo.Input("quality", options=["auto", "low", "medium", "high"],
                                           default="auto", tooltip="Качество генерации gpt-image. auto — не отправлять параметр.")]
        else:
            params = [comfy_io.Combo.Input("aspect_ratio", options=["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "21:9"],
                                           default="auto"),
                      comfy_io.Combo.Input("resolution", options=["auto", "1K", "2K", "4K"], default="auto")]
        options.append(comfy_io.DynamicCombo.Option(m, params))
    return options


class LLMImage(comfy_io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
            node_id="LLMImage",
            display_name="LLM Image",
            category="LLM",
            inputs=[
                comfy_io.DynamicCombo.Input("model", options=_image_model_options()),
                comfy_io.String.Input("prompt", multiline=True, default=""),
                comfy_io.Image.Input("image_1", optional=True,
                                     tooltip="Опционально: референс/исходник — включает режим редактирования (image-to-image)."),
                comfy_io.Image.Input("image_2", optional=True,
                                     tooltip="Опционально: второй референс (совмещение, перенос стиля)."),
                comfy_io.Int.Input("seed", default=0, min=0, max=2**31 - 1, optional=True,
                                   control_after_generate=True,
                                   tooltip="В API не уходит: смени (или randomize), чтобы перегенерировать с теми же входами."),
                comfy_io.String.Input("project", default="", optional=True,
                                      tooltip="Имя проекта для учёта расходов (тег в дашборде). Можно оставить пустым."),
            ],
            outputs=[comfy_io.Image.Output()],
        )

    @classmethod
    def execute(cls, model, prompt, image_1=None, image_2=None, seed=0, project="") -> comfy_io.NodeOutput:
        # model — словарь DynamicCombo: сама модель + параметры её набора
        # seed — только обход кэша ComfyUI, в запрос не идёт
        refs = [_tensor_to_png(t) for t in (image_1, image_2) if t is not None]
        raw = api.image(model["model"], prompt,
                        aspect_ratio=model.get("aspect_ratio", ""),
                        resolution=model.get("resolution", ""),
                        quality=model.get("quality", ""),
                        input_images=refs or None, project=project)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,]
        return comfy_io.NodeOutput(tensor)


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
