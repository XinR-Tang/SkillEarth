"""Tool inferencer package exports.

Keep BaseToolInferencer lazy to avoid an import cycle:
abstract_model -> dynamic_batch_manager -> tool_inferencer.__init__
-> base_inferencer -> abstract_model.
"""

__all__ = ["BaseToolInferencer"]


def __getattr__(name):
    if name == "BaseToolInferencer":
        from .base_inferencer import BaseToolInferencer

        return BaseToolInferencer
    raise AttributeError(name)
