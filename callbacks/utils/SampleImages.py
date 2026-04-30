import copy
from typing import Any, Dict, Optional, Union

from farmhash import Fingerprint64
from torch.utils.data import DataLoader, Dataset, Subset


class SampleImages:
    """
    Hash sampling of image data and wrangling of image metadata and paths.
    """

    def __init__(
        self,
        datastruct: Union[Dataset, DataLoader],
        image_fraction: float,
        dataset_idxs: Optional[list[int]] = None,
    ) -> None:
        """Configure deterministic sample selection from a dataset or dataloader.

        Args:
            datastruct: Dataset or dataloader to sample from.
            image_fraction: Fraction of samples to keep via hash thresholding.
            dataset_idxs: Optional subset indices when ``datastruct`` is a dataset.

        Raises:
            ValueError: If ``dataset_idxs`` is provided when ``datastruct`` is a dataloader.
            TypeError: If ``datastruct`` is not a dataset or dataloader.
        """

        self.datastruct = datastruct
        self.dataset_idxs = dataset_idxs
        self.divisor = 10**6

        if isinstance(datastruct, Dataset):
            self.is_dataset = True
            self.base_dataset = datastruct
        elif isinstance(datastruct, DataLoader):
            self.is_dataset = False
            self.base_dataset = datastruct.dataset.dataset
            if dataset_idxs is not None:
                raise ValueError(
                    "dataset_idxs is only supported when datastruct is a Dataset"
                )
        else:
            raise TypeError(
                f"datastruct must be Dataset or DataLoader, got {type(datastruct).__name__}"
            )

        image_fraction = min(1.0, image_fraction)

        self.upper_thresh = int(self.divisor * image_fraction)

    def sample_images(
        self, metadata_dataset_id: int, metadata_sample_id: Optional[str]
    ) -> Optional[int]:
        """Select a dataset index based on hash-thresholded sample ID.

        Args:
            metadata_dataset_id: Dataset index associated with one sample.
            metadata_sample_id: Sample identifier used for hash-based selection.

        Returns:
            The dataset index when selected, otherwise ``None``.
        """

        if metadata_sample_id is None:
            return metadata_sample_id

        # Samples in the batch may have different IDs
        image_num_id = Fingerprint64(metadata_sample_id) % self.divisor
        if not image_num_id < self.upper_thresh:
            return None

        return metadata_dataset_id

    def __call__(self) -> list[int]:
        """Collect dataset indices selected by hash-based sampling.

        Returns:
            List of selected dataset indices.

        Raises:
            ValueError: If no samples satisfy the sampling fraction.
        """

        image_dataset_idxs = []
        original_data_split = self.base_dataset.split_data
        self.base_dataset.split_data = True
        try:
            data_iterable = (
                (self.datastruct[idx] for idx in self.dataset_idxs)
                if self.is_dataset and self.dataset_idxs is not None
                else self.datastruct
            )

            for data in data_iterable:
                metadata = data["metadata"]
                dataset_ids = metadata["Metadata_Dataset_ID"]
                sample_ids = metadata["Metadata_Sample_ID"]

                dataset_sample_pairs = (
                    [(dataset_ids, sample_ids)]
                    if self.is_dataset
                    else zip(dataset_ids, sample_ids)
                )

                for dataset_id, sample_id in dataset_sample_pairs:
                    image_dataset_idx = self.sample_images(dataset_id, sample_id)

                    if image_dataset_idx is not None:
                        image_dataset_idxs.append(image_dataset_idx)
        finally:
            self.base_dataset.split_data = original_data_split

        if not image_dataset_idxs:
            raise ValueError(
                "No images were sampled. Consider changing your thresholds and ensuring there is enough data in the dataloader."
            )

        return image_dataset_idxs
