# Training the 2d fnet model
The [2d fnet architecture](https://doi.org/10.1038/s41592-018-0111-2) created by Ounkomol et. al. was trained to predict stains from brightfield images.
This task is similar to our task, in which predict cropped nuclear GOLD images from the corresponding DAPI images.
Therefore, we decided to use the 2d fnet model's architecture for our task without pre-training.

To mimic approach of Ounkomol et. al. we train the 2d fnet model using the same hyperparameters.
Similar to Ounkomol et. al., we standardized pixel intensities in each image using the standard scaler transform.
However to ensure the dimensionality was conserved we cropped one pixel from each side of the cropped nuclei images.
