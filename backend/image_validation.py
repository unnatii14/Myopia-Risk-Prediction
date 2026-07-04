"""
Shared fundus-image validation. Used by both the /predict-image endpoint and
the /contribute-image endpoint so uploads are gated consistently.
"""
import numpy as np


def validate_fundus_image(pil_img):
    """
    Heuristic gate: accept only images that look like blue-channel fundus
    exports (a greyscale retinal disc on a dark surround).
    Returns (ok: bool, reason: str | None).
    """
    img = pil_img.convert("RGB").resize((256, 256))
    a = np.asarray(img, dtype=np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    r_mean, g_mean, b_mean = float(r.mean()), float(g.mean()), float(b.mean())

    # 1. Colour profile — training images are blue-channel exports:
    #    either greyscale (R = G = B) or blue-tinted (B > G > R).
    colour_spread = (float(np.mean(np.abs(r - g))) + float(np.mean(np.abs(g - b)))) / 2.0
    is_greyscale = colour_spread <= 12.0
    is_blue_dominant = b_mean > g_mean + 5.0 and g_mean > r_mean - 5.0 and b_mean > r_mean + 15.0
    if not (is_greyscale or is_blue_dominant):
        if r_mean > b_mean + 15.0:
            return False, (
                "This looks like a standard colour (orange/red) fundus photo. "
                "The model needs the blue-channel export of the scan — ask your "
                "eye doctor for it, or use images from the linked sample dataset."
            )
        return False, (
            "This doesn't look like a blue-channel fundus image. The model needs "
            "a dark, blue/greyscale retinal scan — not a colour photo or selfie."
        )

    grey = a.mean(axis=2)

    # 2. Blank / flat image
    if float(grey.std()) < 15.0:
        return False, "The image looks blank or has too little detail to be a retinal scan."

    # 3. Fundus geometry — dark corners around a brighter circular retina
    k = 38  # ~15% of each side
    corners = np.concatenate([
        grey[:k, :k].ravel(), grey[:k, -k:].ravel(),
        grey[-k:, :k].ravel(), grey[-k:, -k:].ravel(),
    ])
    corner_mean = float(corners.mean())
    centre_mean = float(grey[64:192, 64:192].mean())
    if corner_mean > 70.0:
        return False, (
            "No dark surround detected. Fundus scans show a bright circular retina "
            "on a black background — this image doesn't match that pattern."
        )
    if centre_mean < corner_mean + 15.0:
        return False, (
            "The image centre isn't brighter than its edges, which doesn't match "
            "a retinal scan."
        )

    return True, None
