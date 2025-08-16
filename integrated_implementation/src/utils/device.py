"""
Device Management Utilities
==========================

Functions for managing compute devices (CPU/GPU/MPS).
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device(device_str: str = "auto") -> torch.device:
    """
    Get the appropriate device for computation
    
    Args:
        device_str: Device specification ("auto", "cpu", "cuda", "mps")
    
    Returns:
        torch.device object
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
    else:
        device = torch.device(device_str)
        logger.info(f"Using specified device: {device_str}")
    
    return device


def setup_device(device: torch.device) -> torch.device:
    """Setup device and print information"""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        logger.info(f"CUDA Device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif device.type == "mps":
        logger.info("Using Apple MPS acceleration")
    else:
        logger.info("Using CPU for computation")
    
    return device


def get_memory_info(device: torch.device) -> dict:
    """Get memory information for the device"""
    if device.type == "cuda":
        return {
            "allocated": torch.cuda.memory_allocated() / 1e9,
            "reserved": torch.cuda.memory_reserved() / 1e9,
            "total": torch.cuda.get_device_properties(0).total_memory / 1e9
        }
    return {}