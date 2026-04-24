import numpy as np
from MIC import MIC


class ShuffledMIC(MIC):
    """MIC comparator that shuffles paired vectors before scoring."""

    def _preprocess_data(self):
        """Flatten group vectors, then shuffle each to create a null baseline."""

        self._group0, self._group1 = self._group0.iloc[0].values, self._group1.iloc[0].values
        np.random.shuffle(self._group0)
        np.random.shuffle(self._group1)
