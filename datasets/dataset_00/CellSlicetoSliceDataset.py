import pathlib
from typing import Any, Optional

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


class CellSlicetoSliceDataset(Dataset):
    """Dataset over cached DAPI-to-Gold crop pairs."""

    def __init__(
        self,
        manifest_rows: list[dict[str, str]],
        image_specs: dict[str, Any],
        image_preprocessor: Any,
        image_cache_path: Optional[pathlib.Path] = None,
        input_image_name: str = "input_image.tiff",
        target_image_name: str = "target_image.tiff",
    ):
        if not manifest_rows:
            raise ValueError("manifest_rows cannot be empty")

        self.samples = manifest_rows
        self.image_preprocessor = image_preprocessor
        self.image_preprocessor.set_image_specs(**image_specs)
        self.image_cache_path = image_cache_path
        self.input_image_name = input_image_name
        self.target_image_name = target_image_name
        self.device = self.image_preprocessor.device
        self.split_data = False
        self.processing_data = None

    def __len__(self):
        return len(self.samples)

    def _set_metadata(self, idx: int) -> None:
        sample = self.samples[idx]

        self.dataset_id = idx
        self.sample_id = sample["sample_id"]
        self.id = sample["group_id"]

        self.plate = sample.get("plate", "unknown_plate")
        self.well = sample.get("well", "unknown_well")
        self.site = sample.get("site", "unknown_site")
        self.input_channel = sample.get("input_channel", "unknown_channel")
        self.target_channel = sample.get("target_channel", "unknown_channel")

        self.input_path = pathlib.Path(sample["input_path"])
        self.target_path = pathlib.Path(sample["target_path"])

    @property
    def metadata(self):
        return {
            "Metadata_Plate": self.plate,
            "Metadata_Well": self.well,
            "Metadata_Site": self.site,
            "Metadata_Input_Channel": self.input_channel,
            "Metadata_Target_Channel": self.target_channel,
            "Metadata_ID": self.id,
            "Metadata_Dataset_ID": self.dataset_id,
            "Metadata_Sample_ID": self.sample_id,
        }

    def process_load_images(self) -> tuple[torch.Tensor, torch.Tensor]:
        input_image = tifffile.imread(self.input_path).astype(np.float32)
        target_image = tifffile.imread(self.target_path).astype(np.float32)

        self.processing_data = self.image_preprocessor(
            input_img=input_image,
            target_img=target_image,
        )
        return (
            self.processing_data.pop("input_image"),
            self.processing_data.pop("target_image"),
        )

    def __getitem__(self, idx: int):
        self.processing_data = None
        self._set_metadata(idx=idx)

        if self.split_data:
            return {
                "metadata": self.metadata,
                "input_path": self.input_path,
                "target_path": self.target_path,
            }

        cache_dir = None
        if self.image_cache_path is not None:
            cache_dir = self.image_cache_path / self.sample_id

        if cache_dir is not None and cache_dir.exists():
            input_image = torch.from_numpy(
                tifffile.imread(cache_dir / self.input_image_name)
            ).to(device=self.device)
            target_image = torch.from_numpy(
                tifffile.imread(cache_dir / self.target_image_name)
            ).to(device=self.device)
        else:
            input_image, target_image = self.process_load_images()
            if cache_dir is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(cache_dir / self.input_image_name, input_image.cpu().numpy())
                tifffile.imwrite(
                    cache_dir / self.target_image_name, target_image.cpu().numpy()
                )

        return {
            "input": input_image,
            "target": target_image,
            "metadata": self.metadata,
            "processing_metadata": (
                {} if self.processing_data is None else self.processing_data
            ),
            "input_path": self.input_path,
            "target_path": self.target_path,
        }
