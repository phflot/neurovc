import numpy as np
import pytest

from neurovc.momag.framewarpers import (
    TorchForwardSplatWarper,
    forward_splat,
    softmax_splat,
)

torch = pytest.importorskip("torch")


def test_forward_splat_identity_tensor():
    image = torch.arange(9, dtype=torch.float32).view(1, 1, 3, 3)
    flow = torch.zeros(1, 2, 3, 3, dtype=torch.float32)

    warped = forward_splat(image, flow)

    assert warped.shape == image.shape
    assert warped.dtype == image.dtype
    assert torch.allclose(warped, image)


def test_forward_splat_identity_numpy():
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    flow = np.zeros((4, 4, 2), dtype=np.float32)

    warped = forward_splat(image, flow)

    assert isinstance(warped, np.ndarray)
    np.testing.assert_allclose(warped, image)


def test_forward_splat_respects_weight_map():
    image = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    flow = torch.zeros(1, 2, 2, 2)
    weight = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])

    warped = forward_splat(image, flow, weight=weight)

    expected = torch.tensor([[[[0.0, 2.0], [3.0, 0.0]]]])
    assert torch.allclose(warped, expected)


def test_forward_splat_preserves_dtype():
    image = torch.arange(4, dtype=torch.float64).view(1, 1, 2, 2)
    flow = torch.zeros(1, 2, 2, 2, dtype=torch.float64)

    warped = forward_splat(image, flow)

    assert warped.dtype == torch.float64


def test_softmax_splat_matches_forward_for_zero_importance():
    image = torch.rand(1, 3, 4, 5, dtype=torch.float32)
    flow = torch.zeros(1, 2, 4, 5, dtype=torch.float32)
    importance = torch.zeros(1, 1, 4, 5, dtype=torch.float32)

    avg_warp = forward_splat(image, flow)
    softmax_warp = softmax_splat(image, flow, importance)

    assert torch.allclose(avg_warp, softmax_warp, atol=1e-6)


def test_softmax_splat_numpy_input():
    image = np.random.rand(3, 5).astype(np.float32)
    flow = np.zeros((3, 5, 2), dtype=np.float32)
    importance = np.zeros((3, 5), dtype=np.float32)

    warped = softmax_splat(image, flow, importance)

    np.testing.assert_allclose(warped, image)


def test_torch_forward_splat_warper_average_mode():
    image = torch.rand(1, 1, 3, 3)
    flow = torch.zeros(1, 2, 3, 3)

    warper = TorchForwardSplatWarper(mode="average")
    warped = warper(image, flow)

    assert torch.allclose(warped, forward_splat(image, flow))


def test_torch_forward_splat_warper_softmax_requires_importance():
    image = torch.rand(1, 1, 2, 2)
    flow = torch.zeros(1, 2, 2, 2)

    warper = TorchForwardSplatWarper(mode="softmax")
    with pytest.raises(ValueError):
        warper(image, flow)


def test_torch_forward_splat_warper_softmax_mode():
    image = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    flow = torch.zeros(1, 2, 2, 2)
    importance = torch.tensor([[[[0.0, -10.0], [-10.0, -10.0]]]])

    warper = TorchForwardSplatWarper(mode="softmax", temperature=10.0)
    warped = warper(image, flow, importance=importance)

    assert torch.allclose(warped, image, atol=1e-6)
