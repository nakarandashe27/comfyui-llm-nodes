"""Самопроверка api.py: `python test_api.py` из папки пакета. Без сети, torch и ComfyUI."""
import base64
import os
import sys
import types
from unittest import mock

try:
    import requests  # noqa: F401
except ImportError:  # дев-машина без requests: подменяем модуль до импорта api
    fake = types.ModuleType("requests")
    class RequestException(Exception):
        pass
    fake.RequestException = RequestException
    fake.post = fake.get = None
    sys.modules["requests"] = fake

import api  # noqa: E402


class Resp:
    def __init__(self, status=200, json_data=None, content=b"", text="", headers=None):
        self.status_code = status
        self._json = json_data or {}
        self.content = content
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._json


def img_msg(*datas, content=""):
    """Ответ chat-пути с картинками в message.images (data-URL)."""
    images = [{"type": "image_url", "index": i,
               "image_url": {"url": "data:image/png;base64," + base64.b64encode(d).decode()}}
              for i, d in enumerate(datas)]
    return {"choices": [{"message": {"content": content, "images": images}}]}


ENV = {"LITELLM_BASE_URL": "http://gw/", "LITELLM_API_KEY": "sk-test"}


def test_config():
    with mock.patch.dict(os.environ, ENV):
        assert api.load_config() == ("http://gw", "sk-test")  # rstrip слэша
    with mock.patch.dict(os.environ, {}, clear=True), \
         mock.patch.object(api, "_CONFIG_PATH", "no_such_file.ini"):
        try:
            api.load_config()
            assert False, "должен был упасть ConfigError"
        except api.ConfigError as e:
            assert "config.ini" in str(e)


def test_config_bom():
    # Notepad/PowerShell на Windows пишут UTF-8 с BOM — конфиг обязан читаться
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".ini")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(b"\xef\xbb\xbf" + b"[litellm]\nbase_url = http://gw\napi_key = sk-bom\n")
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(api, "_CONFIG_PATH", path):
            assert api.load_config() == ("http://gw", "sk-bom")
    finally:
        os.unlink(path)


def test_error_text_passthrough():
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.object(api.requests, "post",
                           return_value=Resp(400, text="ExceededBudget: key over max_budget")):
        try:
            api.chat("m", "hi")
            assert False
        except RuntimeError as e:
            assert "400" in str(e) and "ExceededBudget" in str(e)


def test_chat_headers_and_result():
    resp = Resp(200, {"choices": [{"message": {"content": "ok!"}}]},
                headers={"x-litellm-response-cost": "0.0012"})
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.object(api.requests, "post", return_value=resp) as post:
        # кириллица в проекте: тег в теле, не заголовком (заголовки latin-1 — падало)
        out, cost = api.chat("m", "hi", system="sys", project="Реклама")
        assert out == "ok!" and cost == 0.0012
        kw = post.call_args.kwargs
        assert kw["headers"]["Authorization"] == "Bearer sk-test"
        assert kw["json"]["metadata"] == {"tags": ["project:Реклама"]}
        assert kw["json"]["messages"][0] == {"role": "system", "content": "sys"}
    # без project — тега нет
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.object(api.requests, "post", return_value=resp) as post:
        api.chat("m", "hi")
        assert "metadata" not in post.call_args.kwargs["json"]


def test_chat_reasoning_effort():
    resp = Resp(200, {"choices": [{"message": {"content": "ok"}}]})
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.object(api.requests, "post", return_value=resp) as post:
        api.chat("m", "hi", reasoning_effort="high")
        assert post.call_args.kwargs["json"]["reasoning_effort"] == "high"
        assert post.call_args.kwargs["json"]["allowed_openai_params"] == ["reasoning_effort"]
        api.chat("m", "hi", reasoning_effort="off")
        assert "reasoning_effort" not in post.call_args.kwargs["json"]
        assert "allowed_openai_params" not in post.call_args.kwargs["json"]


def test_image_chat_request_body():
    resp = Resp(200, img_msg(b"\x89PNGfake"))
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.object(api.requests, "post", return_value=resp) as post:
        api.image_chat("m", "cat", system="sys", aspect_ratio="16:9", image_size="2K",
                       thinking="HIGH", with_text=True, input_images=[b"r1", b"r2"], seed=7)
        body = post.call_args.kwargs["json"]
        assert post.call_args.args[0].endswith("/v1/chat/completions")
        assert body["modalities"] == ["image", "text"]
        assert body["image_config"] == {"aspect_ratio": "16:9", "image_size": "2K"}
        assert body["reasoning"] == {"effort": "high"}
        assert body["seed"] == 7
        assert set(body["allowed_openai_params"]) == {"modalities", "image_config", "reasoning", "seed"}
        assert body["messages"][0] == {"role": "system", "content": "sys"}
        content = body["messages"][1]["content"]
        assert [p["type"] for p in content] == ["image_url", "image_url", "text"]


def test_image_chat_minimal_body():
    resp = Resp(200, img_msg(b"\x89PNGfake"))
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.object(api.requests, "post", return_value=resp) as post:
        api.image_chat("m", "cat")  # всё по умолчанию: только modalities
        body = post.call_args.kwargs["json"]
        assert body["modalities"] == ["image"]
        for absent in ("image_config", "reasoning", "seed"):
            assert absent not in body
        assert body["allowed_openai_params"] == ["modalities"]
        assert len(body["messages"]) == 1  # без system


def test_image_chat_parse_final_thought_cost():
    final, thought = b"FINALIMG", b"THOUGHTIMG"
    resp = Resp(200, img_msg(thought, final, content="note"),
                headers={"x-litellm-response-cost": "0.0387"})
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.object(api.requests, "post", return_value=resp):
        f, text, th, cost = api.image_chat("m", "cat")
        # финал = последняя картинка, thought = первая (OpenRouter не помечает)
        assert f == final and th == thought and text == "note" and cost == 0.0387
    # одна картинка -> thought нет
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.object(api.requests, "post", return_value=Resp(200, img_msg(final))):
        f, text, th, cost = api.image_chat("m", "cat")
        assert f == final and th is None and cost is None


def test_image_chat_no_image_raises():
    resp = Resp(200, {"choices": [{"message": {"content": "не буду"}}]})
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.object(api.requests, "post", return_value=resp):
        try:
            api.image_chat("m", "cat")
            assert False
        except RuntimeError as e:
            assert "не вернула изображение" in str(e)


def test_list_models():
    groups = {"data": [{"model_group": "nano-banana-pro", "mode": "image_generation"},
                       {"model_group": "nano-banana", "mode": "image_generation"},
                       {"model_group": "veo-3", "mode": "video_generation"},
                       {"model_group": "claude-sonnet-5", "mode": "chat"},
                       # развёртка wildcard (со слэшем) в дропдауны не попадает
                       {"model_group": "openrouter/openai/gpt-4o", "mode": "chat"},
                       {"model_group": "openrouter/*", "mode": None}]}
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.dict(api._MODELS_CACHE, {"at": 0.0, "groups": []}), \
         mock.patch.object(api.requests, "get", return_value=Resp(200, groups)):
        assert api.list_models("image_generation") == ["nano-banana", "nano-banana-pro"]
        assert api.list_models("video_generation") == ["veo-3"]
        assert api.list_models("chat") == ["claude-sonnet-5"]
    # нет конфига -> пустой список, не исключение (нода подставит фолбэк)
    with mock.patch.dict(os.environ, {}, clear=True), \
         mock.patch.object(api, "_CONFIG_PATH", "no_such_file.ini"):
        assert api.list_models("image_generation") == []


def test_chat_vision_content():
    resp = Resp(200, {"choices": [{"message": {"content": "вижу"}}]})
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.object(api.requests, "post", return_value=resp) as post:
        assert api.chat("m", "что на фото?", image_png=b"\x89PNGfake")[0] == "вижу"
        content = post.call_args.kwargs["json"]["messages"][-1]["content"]
        assert content[0] == {"type": "text", "text": "что на фото?"}
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_video_poll_then_download():
    submit = Resp(200, {"id": "vid1"})
    polls = [Resp(200, {"status": "processing"}),
             Resp(200, {"status": "completed"}),
             Resp(200, content=b"MP4DATA")]  # последний — скачивание content
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.object(api.requests, "post", return_value=submit), \
         mock.patch.object(api.requests, "get", side_effect=polls) as get, \
         mock.patch.object(api.time, "sleep"):
        out, cost = api.video("m", "cat", seconds=4)
        assert out == b"MP4DATA"
        assert get.call_args.args[0].endswith("/v1/videos/vid1/content")


def test_video_project_tag_both_paths():
    # JSON-путь: metadata — dict; multipart-путь: metadata — JSON-строка (форма)
    import json
    done = [Resp(200, {"status": "completed"}), Resp(200, content=b"MP4")]
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.object(api.requests, "post", return_value=Resp(200, {"id": "v7"})) as post, \
         mock.patch.object(api.requests, "get", side_effect=done * 2), \
         mock.patch.object(api.time, "sleep"):
        api.video("m", "cat", project="Реклама")
        assert post.call_args.kwargs["json"]["metadata"] == {"tags": ["project:Реклама"]}
        api.video("m", "cat", project="Реклама", input_reference_png=b"\x89PNGfake")
        form = post.call_args.kwargs["data"]
        assert json.loads(form["metadata"]) == {"tags": ["project:Реклама"]}


def test_video_failed_raises():
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.object(api.requests, "post", return_value=Resp(200, {"id": "v2"})), \
         mock.patch.object(api.requests, "get",
                           return_value=Resp(200, {"status": "failed", "error": "nsfw"})), \
         mock.patch.object(api.time, "sleep"):
        try:
            api.video("m", "cat")
            assert False
        except RuntimeError as e:
            assert "failed" in str(e) and "nsfw" in str(e)


def test_list_models_failure_backoff():
    # лежащий шлюз: один неудачный запрос на 5 минут, не на каждый вызов
    boom = api.requests.RequestException("net down")
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.dict(api._MODELS_CACHE, {"at": 0.0, "groups": []}), \
         mock.patch.object(api.requests, "get", side_effect=boom) as get:
        assert api.list_models("chat") == []
        assert api.list_models("chat") == []
        assert get.call_count == 1


def test_video_5xx_poll_retry():
    # моргнувший шлюз посреди поллинга — ретрай, а не потеря оплаченного видео
    polls = [Resp(502, text="Bad Gateway"),
             Resp(200, {"status": "processing"}),
             Resp(200, {"status": "completed"}),
             Resp(200, content=b"MP4DATA")]
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.object(api.requests, "post", return_value=Resp(200, {"id": "v5"})), \
         mock.patch.object(api.requests, "get", side_effect=polls), \
         mock.patch.object(api.time, "sleep"):
        assert api.video("m", "cat")[0] == b"MP4DATA"


def test_video_5xx_poll_limit():
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.object(api.requests, "post", return_value=Resp(200, {"id": "v6"})), \
         mock.patch.object(api.requests, "get",
                           return_value=Resp(502, text="Bad Gateway")) as get, \
         mock.patch.object(api.time, "sleep"):
        try:
            api.video("m", "cat")
            assert False
        except RuntimeError as e:
            assert "502" in str(e)
        assert get.call_count == 4  # 1 + 3 ретрая


def test_video_net_retry_limit():
    boom = api.requests.RequestException("net down")
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.object(api.requests, "post", return_value=Resp(200, {"id": "v3"})), \
         mock.patch.object(api.requests, "get", side_effect=boom) as get, \
         mock.patch.object(api.time, "sleep"):
        try:
            api.video("m", "cat")
            assert False
        except api.requests.RequestException:
            assert get.call_count == 4  # 1 + 3 ретрая


def test_video_timeout():
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.object(api.requests, "post", return_value=Resp(200, {"id": "v4"})), \
         mock.patch.object(api.time, "sleep"):
        try:
            api.video("m", "cat", timeout=-1)
            assert False
        except RuntimeError as e:
            assert "не готово" in str(e)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok -", fn.__name__)
    print("%d checks passed" % len(fns))
