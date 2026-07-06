"""HTTP-слой к LiteLLM-шлюзу. Без torch и ComfyUI — проверяется test_api.py.

Ноды не знают про OpenRouter/Gemini/OpenAI: только OpenAI-совместимый API LiteLLM
(nodes-spec.md §0). Ключ и адрес — из config.ini или env, НЕ из графа (§2).
"""
import base64
import configparser
import os
import time

import requests

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")

NO_KEY_MSG = (
    "Нет ключа LiteLLM: заполни config.ini рядом с пакетом (образец — config.ini.example) "
    "или переменные окружения LITELLM_BASE_URL / LITELLM_API_KEY. Ключ выдаёт админ."
)


class ConfigError(RuntimeError):
    pass


def load_config():
    base_url = os.environ.get("LITELLM_BASE_URL", "")
    api_key = os.environ.get("LITELLM_API_KEY", "")
    if not (base_url and api_key) and os.path.exists(_CONFIG_PATH):
        cfg = configparser.ConfigParser()
        cfg.read(_CONFIG_PATH, encoding="utf-8-sig")  # -sig: Notepad/PowerShell пишут BOM
        base_url = base_url or cfg.get("litellm", "base_url", fallback="")
        api_key = api_key or cfg.get("litellm", "api_key", fallback="")
    if not base_url or not api_key:
        raise ConfigError(NO_KEY_MSG)
    return base_url.rstrip("/"), api_key


_MODELS_CACHE = {"at": 0.0, "groups": []}


def list_models(mode):
    """Модели со шлюза (/model_group/info) с данным mode (image_generation, video_generation).
    Пустой список при недоступном шлюзе/конфиге — вызывающий подставит фолбэк."""
    try:
        base_url, key = load_config()
    except ConfigError:
        return []
    if time.monotonic() - _MODELS_CACHE["at"] > 300:
        try:
            r = requests.get(base_url + "/model_group/info", headers=_headers(key), timeout=5)
            _raise_for_error(r)
            _MODELS_CACHE["groups"] = r.json().get("data", [])
            _MODELS_CACHE["at"] = time.monotonic()
        except Exception:
            pass  # шлюз недоступен — работаем с тем, что есть в кэше
    return sorted(g["model_group"] for g in _MODELS_CACHE["groups"] if g.get("mode") == mode)


def _headers(api_key, project=""):
    h = {"Authorization": "Bearer " + api_key}
    if project:
        # тег проекта — единственный источник проектного разреза в дашборде
        h["x-litellm-tags"] = "project:" + project
    return h


def _raise_for_error(resp):
    # текст LiteLLM отдаём как есть: про бюджет/rpm он самодостаточен (nodes-spec §3)
    if resp.status_code >= 300:
        raise RuntimeError("LiteLLM ответил %s: %s" % (resp.status_code, resp.text[:2000]))


def chat(model, prompt, system="", max_tokens=1024, temperature=1.0, project="",
         image_png=None):
    """image_png — опциональные байты картинки: включает vision-режим (промт+изображение)."""
    base_url, key = load_config()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if image_png is not None:
        data_uri = "data:image/png;base64," + base64.b64encode(image_png).decode()
        content = [{"type": "text", "text": prompt},
                   {"type": "image_url", "image_url": {"url": data_uri}}]
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})
    resp = requests.post(
        base_url + "/v1/chat/completions",
        json={"model": model, "messages": messages,
              "max_tokens": max_tokens, "temperature": temperature},
        headers=_headers(key, project), timeout=300)
    _raise_for_error(resp)
    return resp.json()["choices"][0]["message"]["content"]


def _image_config(aspect_ratio="", resolution=""):
    cfg = {}
    if aspect_ratio and aspect_ratio != "auto":
        cfg["aspect_ratio"] = aspect_ratio
    if resolution and resolution != "auto":
        cfg["image_size"] = resolution
    return cfg


def _extract_image(resp):
    item = resp.json()["data"][0]
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"])
    dl = requests.get(item["url"], timeout=300)
    _raise_for_error(dl)
    return dl.content


def image(model, prompt, aspect_ratio="", resolution="", input_images=None, project=""):
    """Возвращает байты картинки. input_images — список png-байтов референсов:
    с ними идём в /v1/images/edits (image-to-image), без них — в /generations."""
    base_url, key = load_config()
    cfg = _image_config(aspect_ratio, resolution)
    if input_images:
        fields = {"model": model, "prompt": prompt, "n": "1"}
        if cfg:
            import json as _json
            fields["image_config"] = _json.dumps(cfg)
        files = [("image", ("ref_%d.png" % i, png, "image/png"))
                 for i, png in enumerate(input_images)]
        resp = requests.post(base_url + "/v1/images/edits", data=fields, files=files,
                             headers=_headers(key, project), timeout=600)
    else:
        payload = {"model": model, "prompt": prompt, "n": 1}
        if cfg:
            payload["image_config"] = cfg
        resp = requests.post(base_url + "/v1/images/generations", json=payload,
                             headers=_headers(key, project), timeout=300)
    _raise_for_error(resp)
    return _extract_image(resp)


def video(model, prompt, seconds="8", size="", input_reference_png=None, project="",
          poll_interval=10, timeout=15 * 60):
    """Сабмит → поллинг до completed/failed → байты mp4 (nodes-spec §4)."""
    base_url, key = load_config()
    fields = {"model": model, "prompt": prompt, "seconds": str(seconds)}
    if size:
        fields["size"] = size
    if input_reference_png is not None:
        resp = requests.post(base_url + "/v1/videos", data=fields,
                             files={"input_reference": ("reference.png", input_reference_png, "image/png")},
                             headers=_headers(key, project), timeout=120)
    else:
        resp = requests.post(base_url + "/v1/videos", json=fields,
                             headers=_headers(key, project), timeout=120)
    _raise_for_error(resp)
    vid = resp.json()["id"]

    deadline = time.monotonic() + timeout
    net_errors = 0
    while True:
        if time.monotonic() > deadline:
            raise RuntimeError("Видео %s: не готово за %d с — проверь статус позже у админа" % (vid, timeout))
        time.sleep(poll_interval)
        try:
            r = requests.get(base_url + "/v1/videos/" + vid, headers=_headers(key), timeout=60)
        except requests.RequestException:
            net_errors += 1  # ретрай только сетевых ошибок поллинга, не генерации (nodes-spec §3)
            if net_errors > 3:
                raise
            continue
        net_errors = 0
        _raise_for_error(r)
        status = r.json().get("status")
        if status == "completed":
            break
        if status == "failed":
            raise RuntimeError("Видео %s: генерация failed: %s" % (vid, r.json().get("error")))

    content = requests.get(base_url + "/v1/videos/" + vid + "/content",
                           headers=_headers(key), timeout=600)
    _raise_for_error(content)
    return content.content
