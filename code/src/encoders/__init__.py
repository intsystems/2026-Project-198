"""Encoder registry and factory.

All encoder implementations auto-register via @register_encoder decorator.
"""

import os
from typing import Any, Callable, Dict, Type

import torch.nn as nn

from src.utils.registry import import_modules

ENCODER_FACTORY: Dict[str, Type[nn.Module]] = {}


def register_encoder(name: str) -> Callable:
    """Decorator to register an encoder class.

    Args:
        name: Registry key for the encoder.
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        ENCODER_FACTORY[name] = cls
        return cls
    return decorator


def EncoderFactory(name: str, **kwargs: Any) -> nn.Module:
    """Instantiate an encoder by registry name.

    Args:
        name: Registered encoder name.
        **kwargs: Forwarded to encoder constructor.

    Returns:
        Instantiated encoder module.

    Raises:
        ValueError: If encoder name is not registered.
    """
    if name not in ENCODER_FACTORY:
        available = ", ".join(sorted(ENCODER_FACTORY.keys()))
        raise ValueError(f"Unknown encoder '{name}'. Available: {available}")
    return ENCODER_FACTORY[name](**kwargs)


# Auto-import all encoder modules to trigger registration
_modules_dir = os.path.dirname(__file__)
import_modules(_modules_dir, "src.encoders")

__all__ = ["ENCODER_FACTORY", "register_encoder", "EncoderFactory"]
