import pathlib
from collections import defaultdict
from typing import Tuple

from farmhash import Fingerprint64
from torch.utils.data import DataLoader, Dataset, Subset

from datasets.dataset_00.utils.Collator import collator


class HashSplitter:
    """
    Create datasplits by hashing sample metadata IDs
    """

    def __init__(
        self,
        dataset: Dataset,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
    ) -> None:
        """Initialize deterministic hash-based split configuration.

        Args:
            dataset: Dataset that exposes ``Metadata_ID`` in sample metadata.
            train_frac: Fraction of samples assigned to the training split.
            val_frac: Fraction of samples assigned to the validation split.
        """

        dataset.split_data = True
        self.dataset = dataset
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.splits = None

    def split_by_hash(self) -> None:
        """Assign dataset indices to train/val/test splits using hash buckets."""

        divisor = 10**6
        self.upper_train_thresh = int(self.train_frac * divisor)
        self.upper_val_thresh = int(self.val_frac * divisor) + self.upper_train_thresh
        self.splits = defaultdict(list)

        for idx, sample in enumerate(self.dataset):
            sample_id = Fingerprint64(sample["metadata"]["Metadata_ID"]) % divisor
            if sample_id < self.upper_train_thresh:
                self.splits["train"].append(idx)
            elif sample_id < self.upper_val_thresh:
                self.splits["val"].append(idx)
            else:
                self.splits["test"].append(idx)

    def __call__(self, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Build dataloaders for train/validation/test splits.

        Args:
            batch_size: Batch size for all returned dataloaders.

        Returns:
            Tuple of ``(train_loader, val_loader, test_loader)``.
        """

        if self.splits is None:
            self.split_by_hash()

        def make_loader(indices, shuffle):
            """Create a subset dataloader with repo-specific collation."""

            return DataLoader(
                Subset(self.dataset, indices),
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=collator,
            )

        self.dataset.split_data = False
        return (
            make_loader(self.splits["train"], shuffle=True),
            make_loader(self.splits["val"], shuffle=False),
            make_loader(self.splits["test"], shuffle=False),
        )
