"""Metrics for "how closely does a generated image follow the control image?".

Most of these re-extract structure from the generated image and compare it with the Canny map that was
fed to ControlNet. Canny F1 alone is a weak signal here (control maps are dense, so unrelated images
already score about 0.45), which is why the softer and threshold-free metrics are included.

Edge maps are boolean (H, W) arrays, images are float arrays in [0, 1] with shape (H, W, 3).
"""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel


def canny(img_uint8, low=100, high=200):
    """Canny edge map of an RGB uint8 image."""
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, low, high) > 0


def dilate(mask, radius):
    """Grow a boolean mask by `radius` pixels (square neighbourhood)."""
    grown = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            grown |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return grown


def edge_f1(generated, control, tol=2):
    """Edge agreement with a pixel tolerance.

    Precision is the share of generated edge pixels within `tol` px of a control edge, recall is the
    share of control edge pixels within `tol` px of a generated edge. Strict IoU is returned as well.
    """
    generated, control = generated.astype(bool), control.astype(bool)
    if generated.sum() == 0 or control.sum() == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}

    precision = (generated & dilate(control, tol)).sum() / generated.sum()
    recall = (control & dilate(generated, tol)).sum() / control.sum()
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    iou = (generated & control).sum() / (generated | control).sum()
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


def soft_recall(generated, control, sigma=3.0):
    """Distance-based recall of control edges.

    For every control edge pixel we take the distance to the nearest generated edge. `soft_recall` is
    the mean of exp(-d^2 / 2 sigma^2) (1 means every control edge has a generated edge on top of it)
    and `chamfer_px` is the plain mean distance in pixels (lower is better). Extra edges in the
    generated image do not hurt either number, unlike precision.
    """
    generated, control = generated.astype(bool), control.astype(bool)
    if control.sum() == 0:
        return {"soft_recall": float("nan"), "chamfer_px": float("nan")}
    if generated.sum() == 0:
        return {"soft_recall": 0.0, "chamfer_px": float(np.hypot(*control.shape))}

    dist = distance_transform_edt(~generated)[control]
    return {
        "soft_recall": float(np.exp(-(dist**2) / (2 * sigma**2)).mean()),
        "chamfer_px": float(dist.mean()),
    }


def grad_corr(img_a, img_b, sigma=2.0):
    """Correlation of blurred Sobel gradient-magnitude maps of two images.

    Needs no thresholds and no binary edges: it only asks whether strong gradients sit in the same
    places. Returns NaN if either image is flat.
    """

    def gradient_magnitude(img):
        gray = img.astype(np.float64).mean(axis=-1)
        return gaussian_filter(np.hypot(sobel(gray, 0), sobel(gray, 1)), sigma).ravel()

    a, b = gradient_magnitude(img_a), gradient_magnitude(img_b)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def canny_multi_f1(
    img_uint8, control, thresholds=((50, 150), (100, 200), (150, 250)), tol=2
):
    """Edge F1 against the fixed control map for several Canny threshold pairs, plus their mean.

    Shows how much the F1 depends on the detector settings used on the generated image.
    """
    scores = {}
    for low, high in thresholds:
        scores[f"f1_t{low}_{high}"] = edge_f1(
            canny(img_uint8, low, high), control, tol
        )["f1"]
    scores["f1_multi_mean"] = float(np.mean(list(scores.values())))
    return scores


def effect_recovery_img(img, img_full, img_none):
    """How close `img` is to the full-ControlNet image, relative to the no-control image.

    1 means identical to full ControlNet, 0 means as far from it as the no-control image is. Measured as
    1 - |img - full| / |none - full| in pixel space. NaN if full and none are identical.
    """
    img, img_full, img_none = (
        np.asarray(x, dtype=np.float64) for x in (img, img_full, img_none)
    )
    scale = np.linalg.norm(img_none - img_full)
    if scale < 1e-9:
        return float("nan")
    return float(1 - np.linalg.norm(img - img_full) / scale)
