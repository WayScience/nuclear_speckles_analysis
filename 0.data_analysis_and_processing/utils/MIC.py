from collections import defaultdict
from typing import Optional

import pandas as pd
from Comparator import Comparator
from minepy import MINE


class MIC(Comparator):
    """Compute and store MIC-style comparisons between paired grouped profiles.

    In the well-level analysis flow, each input group is expected to contain a
    single aggregated row (one profile vector per grouped well/stain). MIC is
    then computed across feature dimensions of the paired vectors.

    This makes the resulting score most appropriate as a descriptive
    cross-feature profile similarity measure between stains within a group,
    rather than a population-level mutual information estimate across repeated
    observations (for example, wells or cells).
    """

    def __init__(self, _comparison_name: str = "mic_e", _mine_params: Optional[dict[str, str]] = None):
        """Initialize MIC comparison storage and MINE configuration.

        Args:
            _comparison_name: Output key used to store computed MIC values.
            _mine_params: Optional keyword arguments forwarded to ``minepy.MINE``.
        """

        self._comparisons = defaultdict(list)

        self._comparison_name = _comparison_name

        if _mine_params:
            self._mine_params = _mine_params

        else:
            self._mine_params = {
                "est": "mic_e"
            }

    def save_groups(self, _group_cols: list[str], **_groups: dict[str, pd.DataFrame]):
        """Save column values defining comparison groups"""

        for idx, col in enumerate(_group_cols):
            for group_name, group in _groups.items():
                if len(_group_cols) > 1:
                    self._comparisons[f"{col}__{group_name}"].append(group[idx])
                else:
                    self._comparisons[f"{col}__{group_name}"].append(group)

    def _preprocess_data(self):
        """Flatten one-row grouped DataFrames into vectors for MINE scoring.

        Each group is expected to have one aggregated row; the first row is used
        intentionally to convert that profile into a 1D feature vector.
        """

        self._group0, self._group1 = self._group0.iloc[0].values, self._group1.iloc[0].values

    @property
    def comparisons(self):
        """Return the accumulated comparison results."""

        return self._comparisons

    def __call__(self, _group0: pd.DataFrame, _group1: pd.DataFrame):
        """Compute a MIC score between two grouped profile vectors.

        Notes:
            Inputs are grouped DataFrames representing paired profiles.
            In the common well-level workflow, each group has one aggregated row
            and MIC is computed across matched feature dimensions. The resulting
            score is descriptive and should be interpreted as cross-feature
            profile similarity, not as population-level mutual information.
        """

        self._group0, self._group1 = _group0, _group1
        self._preprocess_data()

        mine = MINE(**self._mine_params)

        mine.compute_score(self._group0, self._group1)
        self._comparisons[self._comparison_name].append(mine.mic())
