import numpy as np
from exp_utils import *

def do_convert(kspaces, acc, num_iters=200):
    print("acc: ", acc)
    freq_dict={4:25, 8:16, 16:12}
    num_low_freqs = freq_dict[acc]
    pred=[]
    for kspace in kspaces:
        kspace=kspace.astype(np.complex64)[...,None]
        sens_maps = bart.bart(1, f"ecalib -d0 -m1 -r {num_low_freqs}", kspace)
        # reg_wt=0.01
        # #TV
        # p = bart.bart(
        #     1, f"pics -d0 -S -R T:7:0:{reg_wt} -i {num_iters}", kspace, sens_maps
        # )[0]
        ##
        # L1 wavelet, see help `bart pics -h` and `bart pics -Rh`
        # bart pics -l1 -r0.001 kspace sensitivities image_out
        p = bart(
            1, f"pics -l1 -r0.001 -i {num_iters}", kspace, sens_maps
        )[0]

        pred.append(np.abs(p))

    return np.stack(pred,axis=0)

def convert(imgs, masks):
    _kspace = to_space(imgs)
    acc_dict = [4, 8, 16]
    rt=[]
    for i in range(6):
        id=np.floor(i/2).astype(np.uint)
        acc = acc_dict[id]
        mask = masks[i][None,None]
        pred = do_convert(_kspace*mask, acc)
        rt.append(pred)
    rt = np.stack(rt,axis=1)
    return rt

f = np.load("/home/uqwjian7/workspace/fastMRI/testset_selected.npz")
masks = f['masks'].astype(np.int32)
knees=convert(f['knees'],masks)
brains=convert(f['brains'],masks)
np.savez("results_cs", knees=knees,brains=brains)

