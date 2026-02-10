# pruning/__init__.py
"""
Mistral/LLaMA Layer Pruning Package

Based on "The Unreasonable Ineffectiveness of the Deeper Layers" (ICLR 2025)
"""

from .identity import PassLayer, LlamaPassLayer, MistralPassLayer
from .simdrop import (
    choose_block_to_drop,
    drop_consecutive_layers,
    _detect_model_type,
    _get_layers,
)
from .bundler import (
    export_layer_bundle,
    export_two_bundles,
    split_indices,
)

__version__ = "1.0.0"
__all__ = [
    # PassLayer
    "PassLayer",
    "LlamaPassLayer",
    "MistralPassLayer",
    # Pruning
    "choose_block_to_drop",
    "drop_consecutive_layers",
    # Bundler
    "export_layer_bundle",
    "export_two_bundles",
    "split_indices",
    # Utils
    "_detect_model_type",
    "_get_layers",
]