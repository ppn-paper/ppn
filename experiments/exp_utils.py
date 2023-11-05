import matplotlib.pyplot as plt
import numpy as np
import h5py
import sigpy as sp
from scipy.ndimage import zoom
import sigpy.mri as mr
from einops import rearrange
import os
import glob
import torch  as th
import nibabel as nib
# from bart.bart import bart
import multiprocessing
import torch
from sigpy.mri import poisson
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
import shutil
import mat73


def ssim(img1, img2, max=1):
    return ssim_fn(np.abs(img1).squeeze(), np.abs(img2).squeeze(), data_range=max)

def psnr(img1, img2, max=1):
    return psnr_fn(np.abs(img1).squeeze(), np.abs(img2).squeeze(), data_range=max)

def ssim_norm(img1, img2, max=1):
    img1 /= img1.max()
    img2 /= img2.max()
    return ssim_fn(np.abs(img1).squeeze(), np.abs(img2).squeeze(), data_range=max)

def psnr_norm(img1, img2, max=1):
    img1 /= img1.max()
    img2 /= img2.max()
    return psnr_fn(np.abs(img1).squeeze(), np.abs(img2).squeeze(), data_range=max)


def list_file(path):
    files = glob.glob(path)
    for file in files:
        yield file


def compress_folder(folder_path, output_filename):
    # The output filename should have the .zip extension
    shutil.make_archive(output_filename, 'zip', folder_path)
    print(f'{output_filename}.zip created successfully')



def showSamples(fn=None, shape=(2,2), size=5, sel_ids=None, showtest=False, need_rss=False, need_rt=False):
    if not fn:
        directory = '/home/uqwjian7/workspace/ppn_isbi/working/sampling'
        files = glob.glob(os.path.join(directory, '*.nii'))
        if not showtest:
            files = [f for f in files if 'testset.nii' not in f]
            fn = max(files, key=os.path.getmtime)
        else:
            fn="/home/uqwjian7/workspace/ppn_isbi/working/sampling/testset.nii"
        
    print(fn)

    nii_img = nib.load(fn)
    nii_data = nii_img.get_fdata()
    imgs = np.transpose(nii_data, (2, 0, 1))
    if need_rss:
        imgs = rss_complex(imgs)
    if sel_ids is not None:
        imgs = imgs[sel_ids]
    plotimgs(imgs, shape, (size,size))
    if "#" in fn:
        print(fn.split("#")[1])
    
    if need_rt:
        return imgs


def showSamples_np(fn=None, shape=(2,2), need_rss=False):
    if not fn:
        directory = '/home/uqwjian7/workspace/ppn_current/working/sampling'
        files = glob.glob(os.path.join(directory, '*.npz'))
        fn = max(files, key=os.path.getmtime)
    print(fn)
    imgs = np.load(fn)['all_imgs']
    if need_rss:
        imgs = rss_complex(imgs)
        
    plotimgs(imgs, shape)
    if "#" in fn:
        print(fn.split("#")[1])

def plotimgs(imgs, shape=(3,3), figsize=(5,5)):
    print("shape: ", imgs.shape, shape)
    if np.prod(shape)>1:
        fig, axs = plt.subplots(*shape, figsize=figsize)  
        for img, ax in zip(imgs, axs.flatten()):
            ax.imshow(np.abs(img.squeeze()), cmap="gray")
            ax.axis('off')
        plt.tight_layout() 
        plt.show()
    else:
        if len(imgs.shape)==3:
            imgs = imgs[0]
        plt.figure(figsize=figsize)
        plt.imshow(np.abs(imgs).squeeze(), cmap="gray")
        plt.tight_layout()
        plt.show()

def show_Errormap(original_image, generated_image, title="Error Map"):
    # Calculate pixel-wise absolute differences
    error_map = np.abs(original_image - generated_image)

    # Create a density colormap
    cmap = plt.cm.viridis  # Choose a colormap of your preference

    # Plot the error map with a colorbar
    plt.figure(figsize=(4, 3))
    plt.imshow(error_map, cmap=cmap)
    plt.colorbar(label="Error")
    plt.title(title)
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    plt.show()

# def rss_complex(d, dim=-3):  # b 15 w h -> b 1 w h
#     dd = np.stack([d.real, d.imag], axis=-1)  # b 15 w h -> b 15 w h 2
#     dd = np.sqrt((dd**2).sum(axis=-1).sum(axis=dim))
#     return np.expand_dims(dd, axis=dim)


# def rss_complex(d, dim=1):
#     # Assuming d is a numpy array with shape (b, 15, w, h)
#     # Square the real and imaginary parts and sum them up
#     squared_sum = np.sum(np.real(d)**2 + np.imag(d)**2, axis=-1)
    
#     # Sum along the specified dimension and take the square root
#     rss = np.sqrt(np.sum(squared_sum, axis=dim, keepdims=True))
    
#     return rss

def rss_complex(d, dim=1): # complex: (b,c,w,h) => real: (b,1,w,h)
    dd = np.stack([d.real, d.imag], axis=-1)  # Shape changes from (b, 15, w, h) to (b, 15, w, h, 2)s
    return np.sqrt((dd ** 2).sum(axis=-1).sum(axis=dim, keepdims=True))

# def rss_complex(d, axis=-3):  # b c w h -> b 1 w h
#     dd = th.view_as_real(d) # b c w h -> b c w h 2
#     return th.sqrt((dd**2).sum(dim=-1).sum(dim=axis)).unsqueeze(axis)

def rss_real(d, axis=1):  # b 15 w h -> b 1 w h
    return np.sqrt((d**2).sum(axis, keepdims=True))

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

def saveNii(imgs, _fn):
    fn = "%s.nii"%(_fn)
    if not os.path.exists(fn):
        imgs = np.transpose(imgs.numpy().squeeze(), (1, 2, 0))
        imgs = nib.Nifti1Image(imgs, np.eye(4))  # np.eye(4) is an identity affine
        nib.save(imgs, fn)




def print_All(imgs, shape=(1,3), size=4, cmap="gray"):
    if len(imgs)<3:
        shape=(1,len(imgs))
    fig, axs = plt.subplots(*shape, figsize=(size,size))  
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    for img, ax in zip(imgs, axs):
        ax.imshow(np.abs(img.squeeze()), cmap=cmap)
        ax.axis('off')
    plt.tight_layout() 
    plt.show()

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

to_space = lambda imgs, axes=(-2,-1): get_kspace_np(imgs, axes) 
from_space = lambda ks, axes=(-2,-1): kspace_to_image_np(ks, axes) 


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

to_space_th = lambda x: get_kspace(x, (-2, -1)) # x: b c w h
from_space_th = lambda x: kspace_to_image(x, (-2, -1))


def preprocess_mri_images(imgs, new_dim=320, isComplex=True):
    """
    Preprocess a multi-coil MRI image dataset by cropping the center and resizing.

    Parameters:
        dataset (numpy.ndarray): The original dataset. Should be complex-valued with shape (num_images, num_coils, height, width).
        new_dim (int): The height and width for the output images.

    Returns:
        numpy.ndarray: The preprocessed dataset.
    """
    
    # Get the original dimensions
    num_images, num_coils, height, width = imgs.shape
    
    # Determine the smaller dimension and calculate cropping indices
    smaller_dim = min(height, width)
    start_x = (width - smaller_dim) // 2
    start_y = (height - smaller_dim) // 2
    end_x = start_x + smaller_dim
    end_y = start_y + smaller_dim
    
    # Crop the center to make square images
    cropped_set = imgs[:, :, start_y:end_y, start_x:end_x]
    
    # Calculate the scaling factors
    scaling_factors = (1, 1, new_dim / smaller_dim, new_dim / smaller_dim)
    
    if isComplex:
        # Zoom the real and imaginary parts separately
        real_part = zoom(cropped_set.real, scaling_factors)
        imag_part = zoom(cropped_set.imag, scaling_factors)
        
        # Combine back into a complex array
        resized_set = real_part + 1j * imag_part
    else:
        resized_set = zoom(cropped_set, scaling_factors)

    return resized_set

# # prepare complex dataset ===>> not correct!!
# def convert_to_complex_testset(fn, show=False):
#     ff =h5py.File(fn, 'r')
#     rss=ff['reconstruction_rss']

#     ## only take 21%~82% dataset
#     minval, maxval = int(np.ceil(len(rss) * 0.21)), int(np.floor(len(rss) * (1-0.18)))
#     if show:
#         print_All(rss[minval:maxval], shape=(4,5),size=10)
#     kspaces = ff['kspace'][minval:maxval]

#     imgs = from_space(kspaces)
#     imgs = rss_real(imgs.real) + 1j * rss_real(imgs.imag)
#     ii = preprocess_mri_images(imgs)
#     saved_kspace = to_space(ii).astype(np.complex64)
#     return saved_kspace

## prepare the test dataset for multi-coil
def save_mc_testset(fn, save_name, idxs=[15,20,25]):  # multi-coil testset
    with h5py.File(fn, 'r') as files:
        print(list(files))
        print(files['kspace'].shape, files['reconstruction_rss'].shape)
        rss = files['reconstruction_rss'][idxs]
        kspaces = files['kspace'][idxs]
        imgs = from_space(kspaces)
        ii = preprocess_mri_images(imgs)
        saved_kspace = to_space(ii).astype(np.complex64)
        print("---build sensitivity maps---")
        mpss = []
        for ks in saved_kspace:
            # using gpu: device=sp.Device(0)
            # need: pip install cupy-cuda12x
            mps = mr.app.EspiritCalib(ks,device=sp.Device(0)).run()
            mpss.append(mps)
            
        np.savez(save_name, kspace=saved_kspace, rss=rss, sens=np.stack(mpss,axis=0))



def save_complex_for_mc(fn, save_name, idx=[0], coil=20):
    with h5py.File(fn, 'r') as files:
        print(list(files))
        print(files['kspace'].shape, files['reconstruction_rss'].shape)

        kspaces = files['kspace'][idx]
        imgs = from_space(kspaces)
        ii = preprocess_mri_images(imgs)
        saved_kspace = to_space(ii).astype(np.complex64)
        saved_kspace=rearrange(saved_kspace, 'b c h w -> (b c) 1 h w', c=coil)
            
        np.savez(save_name, kspace=saved_kspace)

def create_mc_via_bart(kspace):
    s_maps = np.zeros( kspace.shape, dtype = kspace.dtype)
    for slice_idx in range(kspace.shape[0]):
        gt_ksp = kspace[slice_idx]
        s_maps_ind = bart(1, 'ecalib -m1 -W -c0', gt_ksp.transpose((1, 2, 0))[None,...]).transpose( (3, 1, 2, 0)).squeeze()
        s_maps[slice_idx] = s_maps_ind
    return s_maps

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
    return mask.numpy()



def get_mvue(kspace, s_maps):
    # convert from k-space to mvue
    return np.sum(sp.ifft(kspace, axes=(-1, -2)) * np.conj(s_maps), axis=-3) / np.sqrt(np.sum(np.square(np.abs(s_maps)), axis=-3))

def img_mvue(x, s_maps):
    # convert from image to mvue
    coils = s_maps * x[None, ...] 
    mvue = np.sum(coils * np.conj(s_maps), axis=0) / np.square(np.linalg.norm(s_maps, axis=0))
    return mvue

def img_rss(x, s_maps):
    # convert from image to rss
    coils = s_maps * x[None, ...] 
    rss  = sp.rss(coils, axes=(0,))
    return rss

# refer: https://github.com/utcsilab/csgm-mri-langevin/blob/main/utils.py#L198
# Generate measurements directly from raw fastMRI data files
# Includes rescaling to 384 x 384, ACS-based scaling
# and masking
def get_measurements(raw_file, slice_idx=1, size=320, mask=None):
    if mask == None:
        mask = np.ones((size,size))
    # Load file and get slice
    with h5py.File(raw_file, 'r') as data:
        gt_ksp = np.asarray(data['kspace'][slice_idx])

    # Crop lines in k-space to 384
    gt_ksp = sp.resize(gt_ksp, (
        gt_ksp.shape[0], gt_ksp.shape[1], size))

    # Reduce FoV by half in the readout direction
    gt_ksp = sp.ifft(gt_ksp, axes=(-2,))
    gt_ksp = sp.resize(gt_ksp, (gt_ksp.shape[0], size,
                                gt_ksp.shape[2]))
    gt_ksp = sp.fft(gt_ksp, axes=(-2,)) # Back to k-space

    # ACS-based scaling
    # !!! Change this to pixel-based if desired
    acs          = sp.resize(gt_ksp, (26, 26))
    scale_factor = np.max(np.abs(acs))

    # Downsample and scale
    measured_ksp = gt_ksp * mask[None, None, :]
    measured_ksp = measured_ksp / scale_factor
    gt_ksp       = gt_ksp / scale_factor

    return measured_ksp, gt_ksp, scale_factor

# fn="/mnt/resDisk/Medical/fasterMRI/singlecoil_val/file1000000.h5"
def convert_to_complex_testset(fn, show=False):
    with h5py.File(fn, 'r') as ff:
        kspaces=ff['kspace'][()][:,None]  #(35, 1, 320, 320)
        ii = preprocess_mri_images(from_space(kspaces))  # cut center and scale to (320,320)
        ii /= np.abs(ii).max() #normalize
        return ii.astype(np.complex64)

def run_all_cpu(func, params,n_processes= multiprocessing.cpu_count()):
    print("the number of CPU: ", n_processes)
    with multiprocessing.Pool(n_processes) as pool:
        pool.starmap(func, params)


def run_all_cpu_with_returns(func, params,n_processes= multiprocessing.cpu_count()):
    print("the number of CPU: ", n_processes)
    with multiprocessing.Pool(n_processes) as pool:
        all_results = pool.starmap(func, params)

    # Flatten the list of results and concatenate all images into a single array
    all_imgs = [img for result in all_results for img in result]
    alls = np.concatenate(all_imgs, axis=0)
    return alls
