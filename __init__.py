from .nodes import LLMText, LLMImage, LLMVideo

WEB_DIRECTORY = "./js"  # cost_badge.js: бейдж стоимости последней генерации

NODE_CLASS_MAPPINGS = {
    "LLMText": LLMText,
    "LLMImage": LLMImage,
    "LLMVideo": LLMVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMText": "LLM Text",
    "LLMImage": "LLM Image",
    "LLMVideo": "LLM Video",
}
