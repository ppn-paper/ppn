# PPN project

This is the codebase for [Fast Controllable Diffusion Models for MRI Undersampling Reconstruction]().

This repository is based on [openai/improved-diffusion](https://github.com/openai/improved-diffusion), with modifications for classifier conditioning and architecture improvements.

# Download pre-trained models   

We have released checkpoints for the main models in the paper. Before using these models, please review the corresponding [model card](model-card.md) to understand the intended use and limitations of these models.

Here are the download links for each model checkpoint:

 <!-- * 64x64 classifier: [64x64_classifier.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt)
  -->

# Sampling from pre-trained models

please see the file `.vscode/settings.json`

<!-- To sample from these models, you can use the `classifier_sample.py`, `image_sample.py`, and `super_res_sample.py` scripts.
Here, we provide flags for sampling from all of these models.
We assume that you have downloaded the relevant model checkpoints into a folder called `models/`.

For these examples, we will generate 100 samples with batch size 4. Feel free to change these values.

```
SAMPLE_FLAGS="--batch_size 4 --num_samples 100 --timestep_respacing 250"
``` -->

# Results

the detail, please see the jupyter notebook at folder `experiments`:

## 1. test for real
see `experiments/test_real.ipynb`

### 1.1. Results from Brats

The result is based on a test set containing 64 samples by sampling from last 50 steps from 1000 steps (for x4, x8) or 500 steps (for x16).

| Acceleration          | PNSR  | SSIM |
|------------------|------|-----------|
| x4   | $41.74\pm 2.62$ | $0.983\pm 0.007$      |
| x8 | $37.73\pm 2.55$ | $0.967\pm 0.011$      |
| x16   | $30.25\pm 3.23$ | $0.899\pm 0.036$      |


### 1.2. Results from fastMRI Knee

The result is based on a test set containing 64 samples.

| Acceleration          | PNSR  | SSIM |
|------------------|------|-----------|
| x4   | $36.69\pm 2.51$ | $0.895\pm 0.046$       |
| x8 | $32.65\pm 2.98$ | $0.835\pm 0.072$      |
| x16   | $26.76\pm 4.12$ | $0.707\pm 0.114$      |


### 1.3. Results from fastMRI Brain

The result is based on a test set containing 64 samples.

| Acceleration          | PNSR  | SSIM |
|------------------|------|-----------|
| x4   | $38.52\pm 2.25$ | $0.953\pm 0.021$      |
| x8 | $33.07\pm 2.67$ | $0.914\pm 0.033$      |
| x16   | $26.72\pm 3.30$ | $0.796\pm 0.080$      |


## 2. test for complex
see `experiments/test_complex.ipynb`

## 2.1. Results from Brain (our dataset)

The result is based on a test set containing 8 samples by sampling from last 50 steps from 1000 steps (for x4) or 500 steps (for x8, x16).

| Acceleration          | PNSR  | SSIM |
|------------------|------|-----------|
| x4   | $37.60\pm 3.99$ | $0.958\pm 0.034$      |
| x8 | $30.34\pm 3.19$ | $0.888\pm 0.064$      |
| x16   | $24.15\pm 1.52$ | $0.777\pm 0.058$      |


## 3. test for Multi-Coil
see `experiments/test_multicoil.ipynb`

The result is not satisfactory. The reconstructed result is reconstructed from a subsampled single-coil measurement, taken from multi-coil measurements. I suspect the poor results may be attributed to the dataset used for the model's training. The current training regimen relies on DICOM files, which are fundamentally different from multi-coil data. This discrepancy could be causing the unsucessful result.

I will folllow the paper Jalal, Ajil, et al. "Robust compressed sensing mri with deep generative priors." to retrain the model.

<!-- This table summarizes our ImageNet results for pure guided diffusion models:

| Dataset          | FID  | Precision | Recall |
|------------------|------|-----------|--------|
| ImageNet 64x64   | 2.07 | 0.74      | 0.63   |
| ImageNet 128x128 | 2.97 | 0.78      | 0.59   |
| ImageNet 256x256 | 4.59 | 0.82      | 0.52   |
| ImageNet 512x512 | 7.72 | 0.87      | 0.42   | -->


# Training models
please see the file `.vscode/settings.json`

<!-- Training diffusion models is described in the [parent repository](https://github.com/openai/improved-diffusion). Training a classifier is similar. We assume you have put training hyperparameters into a `TRAIN_FLAGS` variable, and classifier hyperparameters into a `CLASSIFIER_FLAGS` variable. Then you can run:

```
mpiexec -n N python scripts/classifier_train.py --data_dir path/to/imagenet $TRAIN_FLAGS $CLASSIFIER_FLAGS
```

Make sure to divide the batch size in `TRAIN_FLAGS` by the number of MPI processes you are using.

Here are flags for training the 128x128 classifier. You can modify these for training classifiers at other resolutions:

```sh
TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 256 --lr 3e-4 --save_interval 10000 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
``` -->
