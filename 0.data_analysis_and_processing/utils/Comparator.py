from abc import ABC, abstractmethod


class Comparator(ABC):
    """Interface for pairwise comparison backends used in analysis scripts."""

    @abstractmethod
    def save_groups(self, _cols, _drop_cols, _groups):
        """Store grouping metadata associated with a computed comparison."""

        pass

    @abstractmethod
    def _preprocess_data(self):
        """Prepare paired group data before running the comparison metric."""

        pass

    @abstractmethod
    def __call__(self, _group0, _group1):
        """Compute and record a comparison between two grouped data slices."""

        pass
