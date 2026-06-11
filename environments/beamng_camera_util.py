"""Pure camera-frame processing shared by the BeamNG camera environments.

Extracted from environments.beamng so the single-vehicle camera env and the
multi-vehicle env use one implementation. No `self`, no BeamNG connection, no
polling or logging side effects — callers poll the sensor and handle logging.
"""

import numpy as np


def process_camera_frame(colour, out_size) -> np.ndarray:
    """Convert a raw camera ``colour`` frame to a flat grayscale vector in [0, 1].

    Args:
        colour: the camera's ``colour`` channel (PIL image or numpy array,
            RGB/RGBA/grayscale), or None when no frame is available.
        out_size: target (height, width) the frame is resized to.

    Returns:
        A flat float32 array of length ``out_size[0] * out_size[1]`` in [0, 1].
        Returns all-ones (treated as "blank/clear") when ``colour`` is None.
    """
    oh, ow = out_size
    n_pixels = oh * ow
    if colour is None:
        return np.ones(n_pixels, dtype=np.float32)

    # beamngpy may return a PIL Image or a numpy array depending on version.
    img = np.asarray(colour, dtype=np.float32)

    # Convert RGB(A) to grayscale using luminosity weights.
    if img.ndim == 3 and img.shape[2] >= 3:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img.squeeze().astype(np.float32)

    try:
        from PIL import Image as PILImage

        pil = PILImage.fromarray(np.clip(gray, 0, 255).astype(np.uint8), mode="L")
        pil = pil.resize((ow, oh), PILImage.BILINEAR)
        small = np.array(pil, dtype=np.float32)
    except ImportError:
        h, w = gray.shape
        sh, sw = max(1, h // oh), max(1, w // ow)
        small = gray[::sh, ::sw][:oh, :ow]
        if small.shape != (oh, ow):
            padded = np.zeros((oh, ow), dtype=np.float32)
            padded[: small.shape[0], : small.shape[1]] = small
            small = padded

    return (small / 255.0).astype(np.float32).flatten()
