# PPN project

This is the codebase for [Fast Controllable Diffusion Models for Undersampled MRI Reconstruction](https://arxiv.org/abs/2311.12078).

This repository is based on [openai/improved-diffusion](https://github.com/openai/improved-diffusion).

# Download pre-trained models   

We have released checkpoints for the main models in the paper. Here are the download links for each model checkpoint:

 <!-- * 64x64 classifier: [64x64_classifier.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt) -->
 

# Sampling from pre-trained models

The sampling command for BraTS is :

```python -m scripts.image_sample --work_dir working/sampling --model_path evaluations/BraTS/ema_0.9999_600000.pt --testset_path evaluations/BraTS/brats_test.npz --attention_resolutions 30 --class_cond False --learn_sigma True --noise_schedule cosine --image_size 240 --num_channels 32 --num_res_blocks 3 --channel_mult 1,2,2,4,4 --use_ddim True --num_samples 64 --batch_size 32 --timestep_respacing ddim1000 --num_timesteps 50 --acceleration 8 --show_progress True --sampleType PPN```

This command calculates PSNR and SSIM using 64 samples with a batch size of 32. The acceleration factor for the mask is set to 8, and the sample type here is `PPN`. This can be replaced with `DDNM`, `DPS`, or `SONG`, correlating to the `MedScore` method mentioned in the paper. For `PPN`, we utilize the final 50 steps out of 1,000, hence we specify `--timestep_respacing ddim1000 --num_timesteps 50`. Other methods follow the DDIM style for step settings, using only `--timestep_respacing`; for instance, `--timestep_respacing ddim50` implies using 50 DDIM steps. Note that `--num_timesteps` is applicable only for `PPN` and is disregarded by other methods.


# Results

the detail, please see the jupyter notebook at folder `experiments`:

## 1. test for real
see `experiments/test_real.ipynb`

### 1.1. Results from Brats

The result is based on a test set containing 1,000 samples by sampling from last 50 steps from 1000 steps (for x4, x8, x12).

| Acceleration          | PNSR  | SSIM |
|------------------|------|-----------|
| x4   | $41.62\pm 2.83$ | $0.982\pm 0.008$      |
| x8 | $37.51\pm 2.76$ | $0.964\pm 0.012$      |
| x12   | $29.07\pm 3.46$ | $0.902\pm 0.033$      |




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
