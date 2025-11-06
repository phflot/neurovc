import numpy as np
import pytest

from neurovc.momag import flow_processing as fp


def test_get_motion_magnitude_returns_expected_norm():
    flow = np.array([[[3.0, 4.0]]], dtype=np.float32)
    magnitudes = fp.get_motion_magnitude(flow)
    assert magnitudes.shape == (1, 1)
    assert np.allclose(magnitudes[0, 0], 5.0)


def test_compressive_function_thresh_scales_below_threshold():
    g_mag = np.array([[0.5, 1.5]], dtype=np.float32)
    result = fp.compressive_function_thresh(g_mag.copy(), alpha=2.0, threshold=1.0)
    expected = np.array([[1.0, 1.5]], dtype=np.float32)
    assert np.allclose(result, expected)


def test_const_compressor_scales_flow():
    flow = np.ones((1, 1, 2), dtype=np.float32)
    compressor = fp.ConstCompressor(alpha=2.0)
    out = compressor(flow.copy())
    g = np.sqrt(2.0)
    expected_value = (2.0 * g) / (g + 0.0001)
    assert np.allclose(out, expected_value)


def test_thresh_compressor_applies_threshold():
    flow = np.ones((1, 1, 2), dtype=np.float32)
    compressor = fp.ThreshCompressor(alpha=3.0, threshold=2.0)
    out = compressor(flow.copy())
    g = np.sqrt(2.0)
    expected_value = (3.0 * g) / (g + 0.0001)
    assert np.allclose(out, expected_value)


def test_flow_decomposer_creates_mask_and_decomposes_zero_flow():
    landmarks = np.array(
        [
            [8.0, 8.0, 0.0],
            [24.0, 8.0, 0.0],
            [24.0, 24.0, 0.0],
            [8.0, 24.0, 0.0],
        ],
        dtype=np.float32,
    )
    decomposer = fp.FlowDecomposer(landmarks, (32, 32), [0, 1, 2, 3])
    assert decomposer.mask.shape == (32, 32)
    assert decomposer.mask[16, 16] > 0

    flow = np.zeros((32, 32, 2), dtype=np.float32)
    global_flow, local_flow = decomposer.decompose(flow)
    assert np.allclose(global_flow, 0.0)
    assert np.allclose(local_flow, 0.0)


def test_alpha_looper_cycles_values():
    looper = fp.AlphaLooper(alpha=(0.0, 2.0), step=0.5)
    values = [looper() for _ in range(6)]
    assert values == [0.0, 0.5, 1.0, 1.5, 1.0, 0.5]
    looper.reset()
    assert looper() == 0.0


def test_extract_landmarks_success():
    class _DummyLM:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    class _DummyResults:
        def __init__(self):
            self.multi_face_landmarks = [
                type(
                    "LMList",
                    (),
                    {"landmark": [_DummyLM(0.1, 0.2, 0.3), _DummyLM(0.4, 0.5, 0.6)]},
                )
            ]

    result = fp._extract_landmarks(_DummyResults())
    expected = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=float)
    assert np.allclose(result, expected)


def test_extract_landmarks_raises_without_faces():
    with pytest.raises(ValueError):
        fp._extract_landmarks(None)
