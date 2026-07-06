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
    def __init__(self, status=200, json_data=None, content=b"", text=""):
        self.status_code = status
        self._json = json_data or {}
        self.content = content
        self.text = text

    def json(self):
        return self._json


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
    resp = Resp(200, {"choices": [{"message": {"content": "ok!"}}]})
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.object(api.requests, "post", return_value=resp) as post:
        out = api.chat("m", "hi", system="sys", project="ACME")
        assert out == "ok!"
        kw = post.call_args.kwargs
        assert kw["headers"]["Authorization"] == "Bearer sk-test"
        assert kw["headers"]["x-litellm-tags"] == "project:ACME"
        assert kw["json"]["messages"][0] == {"role": "system", "content": "sys"}
    # без project — тега нет
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.object(api.requests, "post", return_value=resp) as post:
        api.chat("m", "hi")
        assert "x-litellm-tags" not in post.call_args.kwargs["headers"]


def test_image_b64():
    png = b"\x89PNGfake"
    resp = Resp(200, {"data": [{"b64_json": base64.b64encode(png).decode()}]})
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.object(api.requests, "post", return_value=resp) as post:
        assert api.image("m", "cat") == png
        assert post.call_args.args[0].endswith("/v1/images/generations")
        assert "image_config" not in post.call_args.kwargs["json"]  # auto -> не шлём


def test_image_config_and_edits():
    png = b"\x89PNGfake"
    resp = Resp(200, {"data": [{"b64_json": base64.b64encode(png).decode()}]})
    # aspect_ratio/resolution уходят в image_config
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.object(api.requests, "post", return_value=resp) as post:
        api.image("m", "cat", aspect_ratio="16:9", resolution="2K")
        cfg = post.call_args.kwargs["json"]["image_config"]
        assert cfg == {"aspect_ratio": "16:9", "image_size": "2K"}
    # с референсами — multipart на /v1/images/edits
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.object(api.requests, "post", return_value=resp) as post:
        api.image("m", "cat", input_images=[b"ref1", b"ref2"])
        assert post.call_args.args[0].endswith("/v1/images/edits")
        assert len(post.call_args.kwargs["files"]) == 2


def test_list_models():
    groups = {"data": [{"model_group": "nano-banana-pro", "mode": "image_generation"},
                       {"model_group": "nano-banana", "mode": "image_generation"},
                       {"model_group": "veo-3", "mode": "video_generation"},
                       {"model_group": "openrouter/*", "mode": None}]}
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.dict(api._MODELS_CACHE, {"at": 0.0, "groups": []}), \
         mock.patch.object(api.requests, "get", return_value=Resp(200, groups)):
        assert api.list_models("image_generation") == ["nano-banana", "nano-banana-pro"]
        assert api.list_models("video_generation") == ["veo-3"]
    # нет конфига -> пустой список, не исключение (нода подставит фолбэк)
    with mock.patch.dict(os.environ, {}, clear=True), \
         mock.patch.object(api, "_CONFIG_PATH", "no_such_file.ini"):
        assert api.list_models("image_generation") == []


def test_chat_vision_content():
    resp = Resp(200, {"choices": [{"message": {"content": "вижу"}}]})
    with mock.patch.dict(os.environ, ENV), \
         mock.patch.object(api.requests, "post", return_value=resp) as post:
        assert api.chat("m", "что на фото?", image_png=b"\x89PNGfake") == "вижу"
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
        out = api.video("m", "cat", seconds=4)
        assert out == b"MP4DATA"
        assert get.call_args.args[0].endswith("/v1/videos/vid1/content")


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
