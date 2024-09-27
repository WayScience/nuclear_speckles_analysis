# Training the 2d fnet model
The [2d fnet architecture](https://doi.org/10.1038/s41592-018-0111-2) created by Ounkomol et. al. was trained to predict stains from brightfield images.
This task is similar to our task, in which predict cropped nuclear GOLD images from the corresponding DAPI images.
Therefore, we decided to use the 2d fnet model's architecture for our task without pre-training.

To mimic approach of Ounkomol et. al. we train the 2d fnet model using the same hyperparameters.
To improve performance in generating stains from the fnet model we min-max normalize pixel intensities in each image.
However to ensure the dimensionality is conserved we crop one pixel from each side of the cropped nuclei images.
