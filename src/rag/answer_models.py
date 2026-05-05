SUPPORTED_ANSWER_MODELS: list[dict[str, object]] = [
    {"id": "deepseek-v4-pro", "label": "DeepSeek V4 Pro", "default": False},
    {"id": "deepseek-v4-flash", "label": "DeepSeek V4 Flash", "default": True},
    {"id": "glm-5", "label": "GLM-5", "default": False},
    {"id": "glm-5.1", "label": "GLM-5.1", "default": False},
    {"id": "kimi-k2.5", "label": "Kimi K2.5", "default": False},
    {"id": "kimi-k2.6", "label": "Kimi K2.6", "default": False},
    {"id": "mimo-v2-pro", "label": "MiMo-V2-Pro", "default": False},
    {"id": "mimo-v2-omni", "label": "MiMo-V2-Omni", "default": False},
    {"id": "mimo-v2.5-pro", "label": "MiMo-V2.5-Pro", "default": False},
    {"id": "mimo-v2.5", "label": "MiMo-V2.5", "default": False},
    {"id": "minimax-m2.5", "label": "MiniMax M2.5", "default": False},
    {"id": "minimax-m2.7", "label": "MiniMax M2.7", "default": False},
    {"id": "qwen3.5-plus", "label": "Qwen3.5 Plus", "default": False},
    {"id": "qwen3.6-plus", "label": "Qwen3.6 Plus", "default": False},
]


def get_supported_answer_models() -> list[dict[str, object]]:
    return SUPPORTED_ANSWER_MODELS
