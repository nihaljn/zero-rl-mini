import logging

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name: str
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the model and tokenizer for the given model name and device.

    Args:
        model_name (str): The name of the model to load.
        device (str): The device to load the model on.

    Returns:
        tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )
    logger.info(f"Loaded model {model_name} on device {model.device}")
    return model, tokenizer


def estimator(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return np.float64(1.0)
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def pass_at_k(results: list[int], ks: list[int]) -> dict:
    n = len(results)
    assert all(k <= n for k in ks), f"k must be <= n, but got {ks} for n={n}"
    assert all(r in [0, 1] for r in results), f"results must be 0 or 1, but got {results}"
    c = sum(results)
    return {f"pass@{k}": estimator(n, c, k).item() for k in ks}
