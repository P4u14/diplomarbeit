from skimage import color
from skimage.util import img_as_float

from atlas.selector.atlas_selector import IAtlasSelector


class BaseAtlasSelector(IAtlasSelector):
    def select_atlases(self, atlases, target_image, n):
        pass

    @staticmethod
    def to_gray(img):
        # RGBA -> RGB
        if img.ndim == 3 and img.shape[2] == 4:
            img = color.rgba2rgb(img)
        # RGB -> Gray
        if img.ndim == 3 and img.shape[2] == 3:
            img = color.rgb2gray(img)
        # To float [0, 1]
        img = img_as_float(img)
        return img