LM_MOUTH = 1 << 0
LM_EYE_LEFT = 1 << 1
LM_EYE_RIGHT = 1 << 2
LM_FOREHEAD = 1 << 3

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0+unknown"
