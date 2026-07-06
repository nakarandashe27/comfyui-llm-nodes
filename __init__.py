from .nodes import LLMText, LLMImage, LLMVideo

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
