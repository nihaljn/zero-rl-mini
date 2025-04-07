import logging

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """
    Set random seed for all libraries to ensure reproducibility.
    
    Args:
        seed (int): Seed value to use
    """
    import random
    import torch
    import os
    
    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set PyTorch's random seed for all devices (CPU, CUDA, MPS)
    torch.manual_seed(seed)
    
    # If CUDA is available, set its seed too
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
        # These settings can improve reproducibility at the cost of performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # If MPS (Metal Performance Shaders for Mac) is available, set its seed
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_model_and_tokenizer(
    model_name: str,
    dtype: type = torch.bfloat16,
    load_in_half: bool = False,
    use_compile: bool = False
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the model and tokenizer for the given model name and device.

    Args:
        model_name (str): The name of the model to load.
        device (str): The device to load the model on.

    Returns:
        tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=dtype,
    )
    if use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            logger.info("Model successfully compiled with torch.compile")
        except Exception as e:
            logger.warning(f"Failed to compile model: {e}")
    if load_in_half:
        model = model.half()
    model.eval()
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
