"""Sanity tests for the architecture and Grad-CAM. Run with: pytest -q"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import build_resscene_cnn, grad_cam  # noqa: E402


def test_parameter_count_matches_notebook():
    # The notebook's model.summary() prints Total params: 1,309,794
    assert build_resscene_cnn().count_params() == 1_309_794


def test_softmax_output():
    model = build_resscene_cnn()
    probs = model.predict(np.random.rand(2, 150, 150, 3).astype("float32"), verbose=0)
    assert probs.shape == (2, 6)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-4)


def test_grad_cam_heatmap_is_normalised():
    model = build_resscene_cnn()
    cam = grad_cam(model, np.random.rand(150, 150, 3), class_index=0)
    assert cam.ndim == 2
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6
