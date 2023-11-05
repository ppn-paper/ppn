# this file is adapted from https://github.com/yang-song/score_inverse_problems/blob/main/cs.py
import numpy as np
import torch as th
import sigpy as sp
import piq
from guided_diffusion import dist_util, logger
from einops import rearrange
import nibabel as nib
import os
import pywt
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
import matplotlib.pyplot as plt

def report_metrics_and_save(args, testset, samples, sens):
    org_samples = samples.clone()
    # handle complex

    testset = try_rss_complex(testset)
    # if sens!=None: #  multi-coil
    #     samples = 
    # else:
    samples = try_rss_complex(samples)

    # TODO normalization is not correct for multicoil
    mm = np.max([1., testset.max(), samples.max()])  #TODO check whether this is valid????
    
    print("== max ==>>>", mm)
    # mask = np.ones(testset.shape) # testset>0
    mask = (testset>0)
    testset = th.clamp(testset, 0., mm) * mask
    samples = th.clamp(samples, 0., mm) * mask
    
    psnr = np.array([piq.psnr(smp[None, ...], tgt[None, ...], data_range=mm).item() 
                for tgt, smp in zip(testset, samples)])
    ssim = np.array([piq.ssim(smp[None, ...], tgt[None, ...], data_range=mm).item() 
                for tgt, smp in zip(testset, samples)])
    
    psnr2=np.array([psnr_fn(smp.squeeze().numpy(), tgt.squeeze().numpy(),data_range=mm) 
                for tgt, smp in zip(testset, samples)])
    ssim2=np.array([ssim_fn(smp.squeeze().numpy(), tgt.squeeze().numpy(),full=True, data_range=mm)[0] 
                for tgt, smp in zip(testset, samples)])
    
    
    def rmse_fn(images1, images2):
        if images1.shape != images2.shape:
            raise ValueError("The two arrays of images must have the same shape.")
        return np.sqrt(np.mean((images1 - images2) ** 2))

    rmse = np.array([rmse_fn(smp[None, ...], tgt[None, ...]) for tgt, smp in zip(testset.numpy(), samples.numpy())])

    report = "samples#%d_x%d_step%d_psnr_%.4f_%.4f_ssim_%.4f_%.4f_rmse_%.4f_%.4f"%(
        args.num_samples, args.acceleration, args.num_timesteps,
        psnr.mean(), psnr.std(), ssim.mean(), ssim.std(), rmse.mean(), rmse.std())
    logger.log("report1: ", report)

    report = "samples#%d_x%d_step%d_psnr_%.4f_%.4f_ssim_%.4f_%.4f_rmse_%.4f_%.4f"%(
        args.num_samples, args.acceleration, args.num_timesteps,
        psnr2.mean(), psnr2.std(), ssim2.mean(), ssim2.std(), rmse.mean(), rmse.std())
    logger.log("report2: ", report)

    def saveNii(imgs, _fn, force=False):
        fn = "%s/%s.nii"%(logger.get_dir(), _fn)
        if not os.path.exists(fn) or force:
            imgs = imgs.numpy().squeeze()
            if len(imgs.shape)==2:
                imgs = imgs[...,None]  # w h b
            else:
                imgs = np.transpose(imgs, (1, 2, 0))  # w h b
            imgs = nib.Nifti1Image(imgs, np.eye(4))  # np.eye(4) is an identity affine
            nib.save(imgs, fn)

    saveNii(testset, "testset", True)
    saveNii(samples, report)
    

    # save_path = "%s/%s.npz"%(logger.get_dir(), report)
    # logger.log("saving to: ", save_path)
    # np.savez(save_path, all_imgs=org_samples)

# load testset
def get_testset_and_mask(args):
    def processfn(ds, mask):
        _processfn = None
        if "BraTS" in args.model_path or "real" in args.sampleType:
            def _processfn(ds):
                imgs = ds['all_imgs'] # b w h
                imgs = imgs[:min(len(imgs), args.num_samples)]
                imgs = (imgs / 255.0).astype(np.float32)
                if len(imgs.shape)==3:
                    imgs=imgs[:,None]
                imgs = th.from_numpy(imgs)
                imgs = normalize(imgs)
                knowns = (to_space(imgs) + to_space(th.rand_like(imgs)* 0.0))*mask
                return imgs, knowns, None, False  # # b 1 w h
        elif 'multicoil' in args.sampleType:
            def _processfn(ds):
                knowns = ds['kspace']
                acs = sp.resize(knowns, (26, 26))
                knowns /= np.max(np.abs(acs))
                knowns=th.from_numpy(knowns).to(th.complex64)
                knowns = knowns[:min(len(knowns), args.num_samples)]

                if len(knowns.shape)==3:
                    knowns = knowns[None]
                


                imgs0=from_space(knowns)

                imgs0 = np.abs(imgs0.real) + 1j * np.abs(imgs0.imag)
                knowns = to_space(imgs0)

                # imgs = imgs[0][None,...] #imgs[:min(len(imgs), args.num_samples)]
                # imgs /= imgs.abs().max()
                knowns = knowns * mask
                
                sens = None
                if 'sens' in ds:
                    sens = ds['sens']  
                    sens = sens[:min(len(sens), args.num_samples)].astype(np.complex64)
                    sens = th.from_numpy(sens)

                imgs = to_mvue(imgs0, sens)
                imgs /= imgs.abs().max()
                return imgs, knowns, sens, True  # # b 1 w h
        elif 'complex' in args.sampleType:
            def _processfn(ds):
                if "kspace" in ds:
                    # sels=[4,5,7,11,12,14]
                    sens=None
                    sens = ds['sens']#[sels]
                    sens = sens[:min(len(sens), args.num_samples)].astype(np.complex64)
                    sens = th.from_numpy(sens)

                    ks = th.from_numpy(ds['kspace']) 
                    # ks = ks[sels]
                    ks = ks[:min(len(ks), args.num_samples)]
                    imgs = from_space(ks)
                    imgs /= np.abs(imgs).max()
                    ks = to_space(imgs)
                    ks = to_space(imgs.real.abs() + 1j * imgs.imag.abs())
                    knowns = ks * mask
                    return imgs, knowns, sens, True
                else:
                    imgs = ds['all_imgs']
                    if len(ds['all_imgs'].shape)==2:
                        imgs = imgs[None,None,...]
                    elif len(ds['all_imgs'].shape)==3:
                        imgs = imgs[None,...] # b w h
                    imgs = imgs[:min(len(imgs), args.num_samples)]
                    imgs = th.from_numpy(imgs)
                    if imgs.is_complex():
                        imgs = imgs.to(th.complex64)
                    # else:
                    else:
                        if imgs.shape[-1]==2:
                            imgs = np.clip(imgs/255.0, -1, 1).astype(np.float32)
                            
                            imgs = th.view_as_complex(imgs)
                            imgs =imgs[:,None]
                        if len(imgs.shape) < 4: 
                            imgs = imgs[:,None]
                    # normlization
                    imgs /= imgs.abs().max()
                    knowns = to_space(imgs) * mask #+ to_space(th.rand_like(imgs)* 0.01) 
                    return imgs, knowns, None, True  # # b 1 w h
        elif 'DPS' in args.sampleType:
            def _processfn(ds):
                knowns = ds['all_imgs'] # b w h
                knowns = knowns[:min(len(knowns), args.num_samples)]
                knowns = th.from_numpy(knowns)

                if len(knowns.shape) < 4: 
                    knowns = knowns[:,None]
                    
                return knowns, knowns, None, False  # # b 1 w h
            
        return _processfn(ds)

    ds = np.load(args.testset_path)
    
    if 'mask' in ds:
        mask = th.tensor(ds['mask'].squeeze()).to(th.long)
    else:  # create mask using acceleration
        mask = get_cartesian_mask(args.image_size, int(args.image_size/args.acceleration))
        # mask = get_superres_mask(args.image_size, 64)
        
    mask = mask[None, None]  # (1,1,w,h)
    imgs, knowns, sens, isComplex = processfn(ds, mask)

    return imgs, knowns, sens, isComplex, mask

def iter_testset(args, all_knowns, all_sens):
    num_batches = int(np.ceil(len(all_knowns) / args.batch_size))

    for batch in range(num_batches):
        start=batch * args.batch_size
        end=min((batch + 1) * args.batch_size, len(all_knowns))
        knowns = all_knowns[start:end]
        sens = all_sens[start:end] if all_sens is not None else None
        yield knowns, sens

def get_cartesian_mask(size, n_keep=30):
    # shape [Tuple]: (H, W)
    center_fraction = n_keep / 1000
    acceleration = size / n_keep

    num_rows, num_cols = size, size
    num_low_freqs = int(round(num_cols * center_fraction))

    # create the mask
    mask = th.zeros((num_rows, num_cols), dtype=th.float32)
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[:, pad: pad + num_low_freqs] = True

    # determine acceleration rate by adjusting for the number of low frequencies
    adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (
        num_low_freqs * acceleration - num_cols
    )

    offset = round(adjusted_accel) // 2

    accel_samples = th.arange(offset, num_cols - 1, adjusted_accel)
    accel_samples = th.round(accel_samples).to(th.long)
    mask[:, accel_samples] = True

    # print("====>>>>> acc: %.5f, adjust: %.5f" % (acceleration, adjusted_accel))
    return mask

def get_superres_mask(size, n_keep):
    mask = np.zeros((size, size), dtype=np.float32)
    m, s = size // 2, n_keep//2
    mask[m-s:m+s, m-s:m+s] = True
    return th.from_numpy(mask) 

def get_kspace(img, axes):
  shape = img.shape[axes[0]]
  return th.fft.fftshift(
      th.fft.fftn(th.fft.ifftshift(
          img, dim=axes
      ), dim=axes),
      dim=axes
  ) / shape


def kspace_to_image(kspace, axes):
  shape = kspace.shape[axes[0]]
  return th.fft.fftshift(
      th.fft.ifftn(th.fft.ifftshift(
          kspace, dim=axes
      ), dim=axes),
      dim=axes
  ) * shape

to_space = lambda x: get_kspace(x, (-2, -1)) # x: b c w hs
from_space = lambda x: kspace_to_image(x, (-2, -1))


def get_kspace_np(imgs, axes):
    return np.fft.fftshift(
        np.fft.fftn(
            np.fft.ifftshift(imgs, axes=axes), 
            axes=axes, norm="ortho"
        ), axes=axes,
    )

def kspace_to_image_np(kspace, axes):
  return np.fft.fftshift(
        np.fft.ifftn(
            np.fft.ifftshift(kspace, axes=axes), 
            axes=axes, norm="ortho"
        ), axes=axes,
    )

to_space_np = lambda imgs, axes=(-2,-1): get_kspace_np(imgs, axes) 
from_space_np = lambda ks, axes=(-2,-1): kspace_to_image_np(ks, axes) 

to_mc = lambda x, _c=15: th.view_as_complex(rearrange(x, '(b c ch) 1 h w -> b c h w ch', c=_c, ch=2).contiguous())
from_mc = lambda x: rearrange(th.view_as_real(x), 'b c h w ch -> (b c ch) 1 h w') 

def get_noisy_known(known, alpha, beta):
    z = th.rand_like(known)
    return alpha * known + beta * to_space(z)

def merge_known_with_mask(x_space, known, mask, coeff=1.):
    return known * mask * coeff + x_space * (1. - mask * coeff)

# def merge_known_with_mask_mc(x_space, known, mask, sens):
#     return known * mask + x_space * (1. - mask )

def rss_real(d, dim=-3):
    return th.sqrt((d**2).sum(dim=dim, keepdim=True))

#th.sqrt((d.abs()**2).sum(dim=dim, keepdim=True))
# RSS: merge multi-coil complex image to single-coil real image
# d is complex number with dimension b 15 w h
# return real numbe with dimension b 1 w h
def rss_complex(d, dim=1):   # complex:  b 15 w h => real: b 1 w h
    dd = th.view_as_real(d)  # b 15 w h -> b 15 w h 2
    return th.sqrt((dd**2).sum(dim=-1).sum(dim=dim, keepdim=True))  

def rss_complex_np(d, dim=1): # complex: (b,c,w,h) => real: (b,1,w,h)
    dd = np.stack([d.real, d.imag], axis=-1)  # Shape changes from (b, 15, w, h) to (b, 15, w, h, 2)s
    return np.sqrt((dd ** 2).sum(axis=-1).sum(axis=dim, keepdims=True))

def esc_complex(d, sens,dim=1): #emulated single-coil esc
    return (d * sens).sum(dim=dim, keepdim=True)

def try_rss_complex(d, dim=1): 
    if d.is_complex():
        return rss_complex(d, dim)
    else:
        return d

def normalize(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= th.min(img)
    img /= th.max(img)
    return img

def normalize_complex(img):
    """ normalizes the magnitude of complex-valued image to range [0, 1] """
    norm_mag_img = normalize(th.abs(img))
    ang_img = th.angle(img)
    return norm_mag_img * th.exp(1j * ang_img)


def total_variation_loss(image):
    """
    Compute Total Variation loss.
    :param image: 4D tensor of shape (B, C, H, W)
    :return: Scalar tensor containing the TV loss
    """
    # Calculate the differences between adjacent pixels
    diff_h = image[:, :, :-1, :] - image[:, :, 1:, :]
    diff_w = image[:, :, :, :-1] - image[:, :, :, 1:]

    # Calculate the total variation loss
    tv_loss = th.sum(th.abs(diff_h)) + th.sum(th.abs(diff_w))

    return tv_loss
### test

def apply_sparse(img, threshold=0.2):
    coeffs = pywt.dwt2(img.cpu(), 'haar')
    cA, (cH, cV, cD) = coeffs

    # Apply sparsity: Zero out small coefficients to keep only strong features
    cA = np.where(np.abs(cA) > threshold, cA, 0)
    cH = np.where(np.abs(cH) > threshold, cH, 0)
    cV = np.where(np.abs(cV) > threshold, cV, 0)
    cD = np.where(np.abs(cD) > threshold, cD, 0)

    # Perform the inverse 2D DWT to reconstruct the image
    reconstructed = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
    return th.from_numpy(reconstructed).to("cuda")


def from_mvue(img_cpx, s_maps): #[1, 1, 320, 320]) -> [1, 20, 320, 320])
    # snorm = th.sqrt(th.sum(th.square(th.abs(s_maps)), dim=1, keepdim=True))
    return (img_cpx * s_maps) if s_maps!=None else img_cpx #/snorm

def to_mvue(coils, s_maps, needsum=True):  #[1, 20, 320, 320]) -> [1, 1, 320, 320])
    if s_maps == None:
        return coils.abs()
    else:
        # coils = from_space(kspace)
        smaps = th.conj(s_maps)
        snorm = th.sqrt(th.sum(th.square(th.abs(s_maps)), dim=1, keepdim=True))
        return coils * smaps if not needsum else th.sum(coils * smaps, dim=1, keepdim=True) #/ snorm 

def save_img(img, name):
    plt.imsave("working/tmp/" + name, img, cmap='gray')
    # pass
    