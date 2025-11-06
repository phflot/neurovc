from . import flow_processing as _flow_processing
from . import framewarpers as _framewarpers

HAS_MODERNGL = _framewarpers.HAS_MODERNGL
HAS_TORCH = _framewarpers.HAS_TORCH

__all__ = [
    "HAS_MODERNGL",
    "HAS_TORCH",
    *_flow_processing.__all__,
    *_framewarpers.__all__,
]

for _name in _flow_processing.__all__:
    globals()[_name] = getattr(_flow_processing, _name)

for _name in _framewarpers.__all__:
    globals()[_name] = getattr(_framewarpers, _name)

del _flow_processing, _framewarpers, _name
