"""Unit tests for environments.beamng_camera_util — pure camera frame processing."""

import numpy as np
import pytest

from environments.beamng_camera_util import process_camera_frame

OUT = (16, 16)
N = OUT[0] * OUT[1]


def test_none_colour_returns_all_ones():
    out = process_camera_frame(None, OUT)
    assert out.shape == (N,)
    assert np.all(out == 1.0)


def test_rgb_frame_is_grayscaled_resized_and_normalized():
    # Solid mid-grey 84x84 RGB frame -> all pixels ~0.5 after /255.
    frame = np.full((84, 84, 3), 128, dtype=np.uint8)
    out = process_camera_frame(frame, OUT)
    assert out.shape == (N,)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.mean() == pytest.approx(128 / 255, abs=0.02)


def test_output_length_tracks_out_size():
    frame = np.zeros((84, 84, 3), dtype=np.uint8)
    out = process_camera_frame(frame, (8, 8))
    assert out.shape == (64,)


def test_rgba_frame_supported():
    frame = np.full((40, 40, 4), 200, dtype=np.uint8)
    out = process_camera_frame(frame, OUT)
    assert out.shape == (N,)
    assert out.mean() == pytest.approx(200 / 255, abs=0.02)
