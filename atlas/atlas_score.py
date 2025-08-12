from dataclasses import dataclass

from skimage import color
from skimage.util import img_as_float

from atlas.atlas import Atlas


@dataclass(frozen=True)
class AtlasScore:
    atlas: Atlas
    score: float

def _to_gray(img):
    # RGBA -> RGB
    if img.ndim == 3 and img.shape[2] == 4:
        img = color.rgba2rgb(img)
    # RGB -> Gray
    if img.ndim == 3 and img.shape[2] == 3:
        img = color.rgb2gray(img)
    # To float [0, 1]
    img = img_as_float(img)
    return img