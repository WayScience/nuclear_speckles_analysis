import pathlib
import tempfile

import mlflow
import numpy as np
import tifffile


def save_image_mlflow(
    image: np.ndarray,
    save_image_path_folder: str,
    image_filename: str,
) -> None:
    """Write an image to a temporary TIFF and upload it to MLflow.

    Args:
        image: Image array to save.
        save_image_path_folder: MLflow artifact folder path.
        image_filename: Artifact filename.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = pathlib.Path(tmp_dir) / image_filename
        tifffile.imwrite(save_path, image)

        mlflow.log_artifact(local_path=save_path, artifact_path=save_image_path_folder)
