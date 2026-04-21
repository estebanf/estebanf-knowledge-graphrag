SUPPORTED_ANSWER_MODELS: list[dict[str, object]] = [
    {"id": "deepseek/deepseek-v3.2", "label": "deepseek/deepseek-v3.2", "default": False},
    {"id": "minimax/minimax-m2.5", "label": "minimax/minimax-m2.5", "default": False},
    {"id": "minimax/minimax-m2.7", "label": "minimax/minimax-m2.7", "default": False},
    {"id": "z-ai/glm-5.1", "label": "z-ai/glm-5.1", "default": False},
    {"id": "moonshotai/kimi-k2.5", "label": "moonshotai/kimi-k2.5", "default": False},
    {"id": "moonshotai/kimi-k2.6", "label": "moonshotai/kimi-k2.6", "default": False},
    {"id": "qwen/qwen3.6-plus", "label": "qwen/qwen3.6-plus", "default": False},
    {"id": "google/gemma-4-31b-it", "label": "google/gemma-4-31b-it", "default": True},
]


def get_supported_answer_models() -> list[dict[str, object]]:
    return SUPPORTED_ANSWER_MODELS
