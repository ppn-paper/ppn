# PPN project

This is the codebase for [Fast Controllable Diffusion Models for Undersampled MRI Reconstruction](https://arxiv.org/abs/2311.12078).

This repository is based on [openai/improved-diffusion](https://github.com/openai/improved-diffusion).

# Requirements  

If you need to reuse your existing `conda` environment, you can install the required packages as follows:

`pip install -r requirements.txt`

If you wish to create a new `conda` environment, you can execute the following command:

`conda env create -f environment.yml -n ppn`

This will create an environment named "ppn", and you can activate your environment with the command:

`conda activate edm`


# Download pre-trained models   

We have released checkpoints and test sets for the main models in the paper, and BraTS is already included in this repository. Here are the complete download links for each model checkpoint and the associated test data:

https://drive.google.com/drive/folders/1kCKPE22m04tuhN_gFrm6LdW2QLHNvhpj?usp=sharing


# Sampling from pre-trained models

The sampling command, adapted from [openai/improved-diffusion](https://github.com/openai/improved-diffusion), is as follows:

1. Sampling command for BraTS is :

```
python -m scripts.image_sample --work_dir working/sampling --model_path evaluations/BraTS/ema_0.9999_600000.pt --testset_path evaluations/BraTS/brats_test.npz --attention_resolutions 30 --class_cond False --learn_sigma True --noise_schedule cosine --image_size 240 --num_channels 32 --num_res_blocks 3 --channel_mult 1,2,2,4,4 --use_ddim True --num_samples 1000 --batch_size 16 --show_progress True --timestep_respacing ddim1000 --num_timesteps 50 --acceleration 8 --sampleType PPN
```


This command computes PSNR and SSIM using 1,000 samples and a batch size of 16 (`--num_samples 1000 --batch_size 16`, you may reduce these values for environmental setup). The acceleration factor for the mask is set to 8 (`--acceleration 8`), and the sample type here is `PPN`. This can be replaced with `DDNM`, `DPS`, or `SONG`, correlating to the `MedScore` method mentioned in the paper. For `PPN`, we utilize the final 50 steps out of 1,000, hence we specify `--timestep_respacing ddim1000 --num_timesteps 50`. Other methods follow the DDIM style for step settings, using only `--timestep_respacing`; for instance, `--timestep_respacing ddim50` implies using 50 DDIM steps. Note that `--num_timesteps` is applicable only for `PPN` and is disregarded by other methods.

2. Sampling command for FastMRI Brain is :

```
python -m scripts.image_sample --work_dir working/sampling --model_path evaluations/fastMRI_brain/ema_0.9999_340000.pt --testset_path evaluations/fastMRI_brain/brain_real_testset.npz --attention_resolutions 40,20 --diffusion_steps 1000 --class_cond False --learn_sigma True --noise_schedule cosine --image_size 320 --num_channels 32 --num_heads 4 --num_res_blocks 3 --resblock_updown True --use_fp16 False --channel_mult 1,2,4,4,6,6 --use_ddim True --num_samples 1000 --batch_size 16 --timestep_respacing ddim1000 --num_timesteps 50 --acceleration 8 --use_scale_shift_norm True --dropout 0.0 --show_progress True --sampleType PPN
```

3. Sampling command for FastMRI Knee is :

```
python -m scripts.image_sample --work_dir working/sampling --model_path evaluations/fastMRI_Knee/ema_0.9999_2400000.pt --testset_path evaluations/fastMRI_Knee/knee_real_testset.npz --attention_resolutions 40,20 --diffusion_steps 1000 --class_cond False --learn_sigma True --noise_schedule cosine --image_size 320 --num_channels 32 --num_heads 4 --num_res_blocks 3 --resblock_updown True --use_fp16 True --channel_mult 1,2,4,4,6,6 --use_ddim True --num_samples 1000 --batch_size 16 --timestep_respacing ddim1000 --num_timesteps 50 --acceleration 8 --use_scale_shift_norm True --dropout 0.0 --show_progress True --sampleType PPN
```



# Results

## 1. test for real
see `experiments/test_real.ipynb`

### 1.1. Results from Brats

The result is based on a test set containing 1,000 samples by sampling from last 50 steps from 1000 steps (for x4, x8, x12).

| Acceleration          | PNSR  | SSIM |
|------------------|------|-----------|
| x4   | $41.62\pm 2.83$ | $0.982\pm 0.008$      |
| x8 | $37.51\pm 2.76$ | $0.964\pm 0.012$      |
| x12   | $29.07\pm 3.46$ | $0.902\pm 0.033$      |


# Citation

```
@article{jiang2023fast,
  title={Fast Controllable Diffusion Models for Undersampled MRI Reconstruction},
  author={Jiang, Wei and Xiong, Zhuang and Liu, Feng and Ye, Nan and Sun, Hongfu},
  journal={arXiv preprint arXiv:2311.12078},
  year={2023}
}
```


<!-- # Training models
please see the file `.vscode/launch.json` -->

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
