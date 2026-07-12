"""HTTP-слой к LiteLLM-шлюзу. Без torch и ComfyUI — проверяется test_api.py.

Ноды не знают про OpenRouter/Gemini/OpenAI: только OpenAI-совместимый API LiteLLM
(nodes-spec.md §0). Ключ и адрес — из config.ini или env, НЕ из графа (§2).
"""
import base64
import configparser
import json
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
        # метку двигаем и при неудаче: иначе лежащий шлюз = 5с таймаут на каждый /object_info
        _MODELS_CACHE["at"] = time.monotonic()
        try:
            r = requests.get(base_url + "/model_group/info", headers=_headers(key), timeout=5)
            _raise_for_error(r)
            _MODELS_CACHE["groups"] = r.json().get("data", [])
        except Exception:
            pass  # шлюз недоступен — стейл-кэш/фолбэки, ретрай через 5 мин
    # кураторские алиасы админа — без слэша (INDEX.md, решение №8: wildcard убран);
    # фильтр по "/" остаётся защитой на случай возврата полных имён на шлюз
    return sorted(g["model_group"] for g in _MODELS_CACHE["groups"]
                  if g.get("mode") == mode and "/" not in g["model_group"])


def _headers(api_key):
    return {"Authorization": "Bearer " + api_key}


def _tag_meta(project):
    """Тег проекта (единственный источник проектного разреза в дашборде) — в теле
    запроса (metadata.tags), не заголовком x-litellm-tags: заголовки HTTP — latin-1,
    кириллица в имени проекта роняла запрос (UnicodeEncodeError)."""
    return {"tags": ["project:" + project]}


def _raise_for_error(resp):
    # текст LiteLLM отдаём как есть: про бюджет/rpm он самодостаточен (nodes-spec §3)
    if resp.status_code >= 300:
        raise RuntimeError("LiteLLM ответил %s: %s" % (resp.status_code, resp.text[:2000]))


def _cost(resp):
    """Фактическая стоимость вызова из заголовка шлюза (nodes-image-v2 §5.2), None если нет."""
    v = resp.headers.get("x-litellm-response-cost")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def chat(model, prompt, system="", max_tokens=1024, temperature=1.0, project="",
         image_png=None, reasoning_effort="off"):
    """Возвращает (text, cost_usd | None).
    image_png — опциональные байты картинки: включает vision-режим (промт+изображение).
    reasoning_effort — глубина размышлений reasoning-моделей; "off" = не отправлять параметр."""
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
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    if reasoning_effort and reasoning_effort != "off":
        payload["reasoning_effort"] = reasoning_effort
        # маппер openrouter в LiteLLM параметра не знает — просим проброс как есть
        payload["allowed_openai_params"] = ["reasoning_effort"]
    if project:
        payload["metadata"] = _tag_meta(project)
    resp = requests.post(base_url + "/v1/chat/completions", json=payload,
                         headers=_headers(key), timeout=300)
    _raise_for_error(resp)
    return resp.json()["choices"][0]["message"]["content"], _cost(resp)


def image_chat(model, prompt, system="", aspect_ratio="", image_size="", thinking="",
               with_text=False, input_images=None, seed=None, project=""):
    """Картинка через chat-путь шлюза — генерация и редактирование одним маршрутом
    (nodes-image-v2 §5.1, проверено живьём для gemini-семейства и mai).
    Возвращает (final_bytes, text, thought_bytes | None, cost_usd | None).

    input_images — список байтов референсов (image_url-парты перед промтом).
    thinking — "MINIMAL"/"HIGH" -> reasoning.effort (только nb-2/lite); "" = не слать.
    seed=None — не отправлять (mai его не знает). OpenRouter не помечает
    thought-парты, поэтому финал = последняя картинка ответа, thought = первая
    при наличии нескольких."""
    base_url, key = load_config()
    content = [{"type": "image_url",
                "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}
               for png in (input_images or [])]
    content.append({"type": "text", "text": prompt})
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    payload = {"model": model, "messages": messages,
               "modalities": ["image", "text"] if with_text else ["image"]}
    allowed = ["modalities"]  # шлюз пропускает нестандартные параметры только по этому списку
    cfg = {}
    if aspect_ratio and aspect_ratio != "auto":
        cfg["aspect_ratio"] = aspect_ratio
    if image_size and image_size != "auto":
        cfg["image_size"] = image_size
    if cfg:
        payload["image_config"] = cfg
        allowed.append("image_config")
    if thinking:
        payload["reasoning"] = {"effort": thinking.lower()}
        allowed.append("reasoning")
    if seed is not None:
        payload["seed"] = seed
        allowed.append("seed")
    payload["allowed_openai_params"] = allowed
    if project:
        payload["metadata"] = _tag_meta(project)

    resp = requests.post(base_url + "/v1/chat/completions", json=payload,
                         headers=_headers(key), timeout=600)
    _raise_for_error(resp)
    msg = resp.json()["choices"][0]["message"]
    datas = []
    for im in (msg.get("images") or []):
        url = (im.get("image_url") or {}).get("url", "")
        if url.startswith("data:"):
            datas.append(base64.b64decode(url.split(",", 1)[1]))
        elif url:
            dl = requests.get(url, timeout=300)
            _raise_for_error(dl)
            datas.append(dl.content)
    if not datas:
        raise RuntimeError("Модель не вернула изображение (отказ или фильтр контента). "
                           "Ответ: %s" % str(msg.get("content"))[:300])
    thought = datas[0] if len(datas) > 1 else None
    return datas[-1], msg.get("content") or "", thought, _cost(resp)


def video(model, prompt, seconds="8", size="", input_reference_png=None, project="",
          poll_interval=10, timeout=15 * 60):
    """Сабмит → поллинг до completed/failed → (байты mp4, cost_usd | None) (nodes-spec §4).
    Стоимость — из заголовка ответа на сабмит (расход списывается при постановке задачи)."""
    base_url, key = load_config()
    fields = {"model": model, "prompt": prompt, "seconds": str(seconds)}
    if size:
        fields["size"] = size
    if input_reference_png is not None:
        if project:
            # multipart: значения формы — строки; шлюз json.loads-ит поле metadata
            fields["metadata"] = json.dumps(_tag_meta(project))
        resp = requests.post(base_url + "/v1/videos", data=fields,
                             files={"input_reference": ("reference.png", input_reference_png, "image/png")},
                             headers=_headers(key), timeout=120)
    else:
        if project:
            fields["metadata"] = _tag_meta(project)
        resp = requests.post(base_url + "/v1/videos", json=fields,
                             headers=_headers(key), timeout=120)
    _raise_for_error(resp)
    vid = resp.json()["id"]
    cost = _cost(resp)

    deadline = time.monotonic() + timeout
    net_errors = 0
    while True:
        if time.monotonic() > deadline:
            raise RuntimeError("Видео %s: не готово за %d с — проверь статус позже у админа" % (vid, timeout))
        time.sleep(poll_interval)
        try:
            r = requests.get(base_url + "/v1/videos/" + vid, headers=_headers(key), timeout=60)
        except requests.RequestException:
            net_errors += 1  # ретрай только поллинга, не самой генерации (nodes-spec §3)
            if net_errors > 3:
                raise
            continue
        if r.status_code >= 500:  # моргнувший шлюз (рестарт litellm) — тоже ретрай, видео уже оплачено
            net_errors += 1
            if net_errors > 3:
                _raise_for_error(r)
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
    return content.content, cost
