"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    add_dict_to_argparser
)
import ppn.ppn_sample_utils as ppn_sample_utils
from ppn.ppn_diffusion import *
from functools import partial

def load_model(args, device):
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    return model, diffusion
    

def main():
    # init
    device = dist_util.dev()
    args = create_argparser().parse_args()

    mixpercent=0
    if args.sampleType.startswith("multicoil"):
        args.mcType = args.sampleType.split("_")[1]
        args.sampleType="multicoil"

    assert args.sampleType in ['DPS', 'PPN', 'DDPM', 'DDIM', 'real', 'complex', 'cpx_superres', 'real_superres', 'multicoil'], "Sample type is not correct"

    dist_util.setup_dist()
    logger.configure(args.work_dir, ["stdout", "tensorboard"])

    logger.log("creating model and diffusion...")
    model, diffusion = load_model(args, device)

    logger.log("sampling...")
    all_samples = []
    all_imgs, all_knowns, all_sens, isComplex, mask = ppn_sample_utils.get_testset_and_mask(args)
    ppn_loop = partial(diffusion.ppn_loop, model=model, isComplex=isComplex, mask=mask, device=device, 
                       sampleType=args.sampleType, mcType=args.mcType, progress=args.show_progress, mixpercent=mixpercent)

    for knowns, sens in ppn_sample_utils.iter_testset(args, all_knowns, all_sens):
        sample = ppn_loop(knowns, sens)
        all_samples.extend([sample.cpu()])

    logger.log("sampling complete")
    all_samples = th.cat(all_samples, dim=0)  #np.concatenate(all_samples, axis=0)
    
    logger.log_snapshot(try_rss_complex(all_samples))
    args.num_timesteps = diffusion.num_timesteps

    ppn_sample_utils.report_metrics_and_save(args, all_imgs, all_samples, all_sens) # psnr and ssim



def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        work_dir="",
        testset_path="",
        acceleration=4,
        show_progress=False,
        num_timesteps=0,
        sampleType="PPN", # PPN, DDIM, DDPM
        mcType="a", # multi-coil type 
        sensmap_path=""
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
