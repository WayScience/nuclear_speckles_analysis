from albumentations import ImageOnlyTransform
import numpy as np

class MinMaxNormalize(ImageOnlyTransform):
    """Min-Max normalize each image"""

    def __init__(self, _normalization_factor, _always_apply=False, _p=0.5):
        super(MinMaxNormalize, self).__init__(_always_apply, _p)
        self.__normalization_factor = _normalization_factor

    @property
    def normalization_factor(self):
        return self.__normalization_factor

    def apply(self, _img, **params):

        if isinstance(_img, np.ndarray):
            return _img / self.normalization_factor

        else:
            raise TypeError("Unsupported image type for transform (Should be a numpy array)")
