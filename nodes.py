"""ComfyUI-обёртки над api.py. Здесь только конвертация тензоров и файлов."""
import asyncio
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


def _bytes_to_tensor(data):
    # PIL, не ручной PNG-парсинг: nb-pro может отдать и JPEG (nodes-image-v2 §5)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,]


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
    async def execute(cls, model, prompt, system="", image=None,
                      max_tokens=1024, temperature=1.0, seed=0, project="") -> comfy_io.NodeOutput:
        # model — словарь DynamicCombo: сама модель + её reasoning_effort
        # seed — только обход кэша ComfyUI (повторный запуск), в запрос не идёт
        png = _tensor_to_png(image) if image is not None else None
        text, cost = await asyncio.to_thread(
            api.chat, model["model"], prompt, system=system, max_tokens=max_tokens,
            temperature=temperature, project=project, image_png=png,
            reasoning_effort=model.get("reasoning_effort", "off"))
        return comfy_io.NodeOutput(text, ui={"cost_usd": [cost]})


# --- LLM Image v2: параметры по семействам — только проверенное (nodes-image-v2.md §5) ---

_AR_BASE = ["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
_AR_FULL = _AR_BASE + ["1:4", "4:1", "1:8", "8:1"]  # экстремальные — только nb-2
_AR_MAI = ["auto", "1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3"]

_IMG_FAMILIES = {
    #        aspect     sizes                       think  max_img sys    seed_api
    "nb":   (_AR_BASE, None,                        False, 3,      True,  True),
    "pro":  (_AR_BASE, ["1K", "2K", "4K"],          False, 14,     True,  True),
    "nb2":  (_AR_FULL, ["0.5K", "1K", "2K", "4K"],  True,  14,     True,  True),
    "lite": (_AR_BASE, None,                        True,  14,     True,  True),
    "mai":  (_AR_MAI,  None,                        False, 1,      False, False),
}


def _img_family(alias):
    # ponytail: семейство по подстроке кураторского алиаса; незнакомые модели
    # получают консервативный набор "nb" (aspect_ratio + 3 референса)
    a = alias.lower()
    if "mai" in a:
        return "mai"
    if "lite" in a:
        return "lite"
    if "banana-2" in a or "3.1" in a:
        return "nb2"
    if "pro" in a:
        return "pro"
    return "nb"


def _image_model_options():
    """Опции DynamicCombo: набор параметров и лимит референсов зависят от модели."""
    models = sorted(set(api.list_models("image_generation")) |
                    {"nano-banana", "nano-banana-pro", "nano-banana-2",
                     "nano-banana-2-lite", "mai-image-2.5"})
    options = []
    for m in models:
        ar, sizes, think, max_img, has_sys, _ = _IMG_FAMILIES[_img_family(m)]
        inputs = [comfy_io.Combo.Input("aspect_ratio", options=ar, default="auto",
                                       tooltip="auto: под референс, без референса — дефолт модели.")]
        if sizes:
            inputs.append(comfy_io.Combo.Input("resolution", options=["auto"] + sizes, default="auto",
                                               tooltip="auto — дефолт модели (1K). Выше разрешение = дороже."))
        if think:
            inputs.append(comfy_io.Combo.Input("thinking_level", options=["MINIMAL", "HIGH"],
                                               default="MINIMAL",
                                               tooltip="HIGH: дольше и дороже, лучше следование сложному промту."))
        inputs.append(comfy_io.Autogrow.Input(
            "images",
            template=comfy_io.Autogrow.TemplateNames(
                comfy_io.Image.Input("image"),
                names=["image_%d" % i for i in range(1, max_img + 1)],
                min=0),
            tooltip="Референсы (image-to-image): подключи image_1 — появится следующий вход. "
                    "У этой модели до %d." % max_img))
        if has_sys:
            inputs.append(comfy_io.Combo.Input(
                "response_modalities", options=["IMAGE", "IMAGE+TEXT"], default="IMAGE",
                advanced=True,
                tooltip="IMAGE+TEXT: модель вернёт и текстовый комментарий (выход text)."))
            inputs.append(comfy_io.String.Input(
                "system_prompt", multiline=True, default="", optional=True, advanced=True,
                tooltip="Фундаментальные инструкции поверх промта (стиль, правила композиции)."))
        options.append(comfy_io.DynamicCombo.Option(m, inputs))
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
                comfy_io.Int.Input("seed", default=0, min=0, max=2**31 - 1, optional=True,
                                   control_after_generate=True,
                                   tooltip="Уходит в API (best-effort повторяемость) и обходит кэш ComfyUI."),
                comfy_io.String.Input("project", default="", optional=True,
                                      tooltip="Имя проекта для учёта расходов (тег в дашборде). Можно оставить пустым."),
            ],
            outputs=[
                comfy_io.Image.Output(),
                comfy_io.String.Output(display_name="text",
                                       tooltip="Текст ответа модели (при IMAGE+TEXT), иначе пусто."),
                comfy_io.Image.Output(display_name="thought_image",
                                      tooltip="Промежуточная картинка размышлений (nb-pro/nb-2, не всегда); "
                                              "если её нет — дубль финальной."),
            ],
        )

    @classmethod
    async def execute(cls, model, prompt, seed=0, project="") -> comfy_io.NodeOutput:
        # async: копии ноды исполняются параллельно (executor гоняет async-ноды конкурентно)
        _, _, _, _, _, seed_api = _IMG_FAMILIES[_img_family(model["model"])]
        refs = [_tensor_to_png(t) for t in (model.get("images") or {}).values() if t is not None]
        final, text, thought, cost = await asyncio.to_thread(
            api.image_chat, model["model"], prompt,
            system=model.get("system_prompt", ""),
            aspect_ratio=model.get("aspect_ratio", ""),
            image_size=model.get("resolution", ""),
            thinking=model.get("thinking_level", ""),
            with_text=model.get("response_modalities") == "IMAGE+TEXT",
            input_images=refs or None,
            seed=seed if seed_api else None,
            project=project)
        t_final = _bytes_to_tensor(final)
        # ponytail: нет thought-картинки -> дублируем финал, downstream не ломается
        t_thought = _bytes_to_tensor(thought) if thought else t_final
        return comfy_io.NodeOutput(t_final, text, t_thought, ui={"cost_usd": [cost]})


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

        data, cost = api.video(model, prompt, seconds=seconds, size=size,
                               input_reference_png=ref_png, project=project)

        import folder_paths  # модуль ComfyUI, доступен только внутри него
        path = os.path.join(folder_paths.get_output_directory(),
                            "llm_video_%d.mp4" % int(time.time()))
        with open(path, "wb") as f:
            f.write(data)
        return {"ui": {"cost_usd": [cost]}, "result": (path,)}
