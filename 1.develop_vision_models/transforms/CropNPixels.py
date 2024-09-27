from albumentations import ImageOnlyTransform
import numpy as np

class CropNPixels(ImageOnlyTransform):
    def __init__(self, _pixel_count=1, _always_apply=False, _p=0.5):
        super(CropNPixels, self).__init__(_always_apply, _p)
        self.pixel_count = _pixel_count

    def apply(self, _img, **params):
        if isinstance(_img, np.ndarray):

            # Ensure we do not crop more pixels than the image size
            if self.pixel_count * 2 >= _img.shape[0] or self.pixel_count * 2 >= _img.shape[1]:
                raise ValueError(f"Cannot crop {self.pixel_count} pixels, image is too small.")

            # Crop pixel_count pixels from each side
            return _img[self.pixel_count:-self.pixel_count, self.pixel_count:-self.pixel_count]

        else:
            raise TypeError("Unsupported image type for transform (Should be a numpy array)")

    def get_transform_init_args_names(self):
        return ["pixel_count", "always_apply", "p"]
