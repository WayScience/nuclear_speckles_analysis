import numpy as np
import torch
import matplotlib.pyplot as plt
import pathlib
from typing import Optional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def format_img(_tensor_img: torch.Tensor, _normalization_factor: int) -> np.ndarray:
    """Reshapes an image and rescales pixel values from the MinMaxNormalize transform."""

    # pass in norm_factor
    #norm_factor = trainer.val_dataset.dataset.input_transform[0].normalization_factor

    return (torch.squeeze(_tensor_img) * _normalization_factor).to(torch.uint16).cpu().numpy()

def visualize_stains(_input: torch.Tensor, _output: torch.Tensor, _target: torch.Tensor, _image_path: pathlib.Path, _title: str, _pretrained_output: Optional[torch.Tensor] = None):

    max_pixel_val = max(np.max(_input), np.max(_target), np.max(_output))
    min_pixel_val = min(np.min(_input), np.min(_target), np.min(_output))

    titles = ['DAPI Image', 'Predicted GOLD Image']

    if _pretrained_output is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = titles + ['Target GOLD Image']
        idx = 2

    else:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        titles = titles + ['Pretrained Predicted GOLD Image', 'Target GOLD Image']
        min_pixel_val = min(min_pixel_val, np.min(_pretrained_output))
        idx = 3

    fig.suptitle(f"{_title}", fontsize=16)

    plt.subplots_adjust(wspace=0.3, hspace=0)  # Adjust `wspace` if titles overlap

    axes[0].imshow(_input, cmap="gray", vmin=min_pixel_val, vmax=max_pixel_val)
    axes[0].axis('off')
    axes[0].set_title(titles[0], fontsize=14)

    axes[1].imshow(_output, cmap="gray", vmin=min_pixel_val, vmax=max_pixel_val)
    axes[1].axis('off')
    axes[1].set_title(titles[1], fontsize=14)

    if _pretrained_output is not None:
        axes[2].imshow(_pretrained_output, cmap="gray", vmin=min_pixel_val, vmax=max_pixel_val)
        axes[2].axis('off')
        axes[2].set_title(titles[2], fontsize=14)

    axes[idx].imshow(_target, cmap="gray", vmin=min_pixel_val, vmax=max_pixel_val)
    axes[idx].axis('off')
    axes[idx].set_title(titles[idx], fontsize=14)

    plt.savefig(_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
