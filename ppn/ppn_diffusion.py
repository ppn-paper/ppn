from guided_diffusion.respace import *
from guided_diffusion.gaussian_diffusion import _extract_into_tensor
from .ppn_sample_utils import *
from tqdm.auto import tqdm
from functools import partial
from skimage.restoration import denoise_tv_chambolle
from einops import rearrange

class PPN_Diffusion(SpacedDiffusion):
    
    def __init__(self, use_timesteps, **kwargs):
        super().__init__(use_timesteps, **kwargs)

    # @th.no_grad()
    def dps(self, model, x_real, t):
        ts = th.tensor([t]  * x_real.shape[0], device=x_real.device)

        x_real.requires_grad_()   
        out = self.p_sample(model, x_real, ts) #self.ddim_sample(model, x, ts)          

        x_0 = from_mvue(to_mc(out['pred_xstart'], self.coilNum),self.sens)
        # calculate the gradient
        diff = self.mask * (self.knowns - to_space(x_0)) # mask * (y - F * x_0)
        norm = th.linalg.norm(diff)  
        norm_grad = th.autograd.grad(outputs=norm, inputs=x_real)[0] # \nabla_x |y - F * x_0|^2
        # fix the gradient

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, ts, x_real.shape)
        x = out['sample'] - 1*norm_grad #* th.sqrt(alpha_bar)  # x_t - grad * hyperparam

        # x_real.detach()
        return x.detach_()
    
    @th.no_grad()
    def ppn(self, model, x_real, t): # 0: x_0, 1: x_t
        ts = th.tensor([t]  * x_real.shape[0], device=x_real.device)
        x_0 = self.p_mean_variance(model, x_real, ts)["pred_xstart"] # predictor
        x_0_hat = self.projector(x_0, t)
        
        # ppn
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, ts, x_real.shape)
        x_real_pre = th.sqrt(alpha_bar_prev) * x_0_hat  + th.sqrt(1-alpha_bar_prev) * th.rand_like(x_0_hat)

        return x_real_pre
    
    def ppn_loop(self, knowns, sens, mask, model, isComplex=False,
                device="cpu", sampleType="PPN", mcType='a', progress=False, mixpercent=0.0):
        
        print("Sampling type: ", sampleType)
        self.mcType = mcType
        self.coilNum = 1 if mcType=='a' else knowns.shape[1]
        self.mask = mask.to(device)
        self.knowns = knowns.to(device)
        self.sens = sens.to(device) if sens is not None else None
        self.mixstepsize = int(self.num_timesteps * mixpercent) 

        sample_fn = None
        if sampleType == "DPS": 
            sample_fn = partial(self._loop_dps, model=model, progress=progress, device=device)
        else:
            def projector_real(x_real, t=0,lmd=1.0):
                x_space = to_space(x_real)
                x_space = merge_known_with_mask(x_space, self.knowns, self.mask, coeff=lmd)
                return from_space(x_space).real
            self.projector = projector_real
            denoise_fns_real={'DDPM': self.ddpm, 'DDIM': self.ddim, 'PPN': self.ppn, 'real': self.ppn}
            sample_fn = partial(self._loop_real, model=model, denoise_fn=denoise_fns_real['PPN'], 
                                progress=progress, device=device)

        return sample_fn()
    
    def _loop_dps(self, model, progress=False, device="cpu"):
        _indices = list(range(self.num_timesteps))[::-1]
        indices = tqdm(_indices) if progress else _indices
        x = th.randn_like(self.knowns.real, device=device)
        for i in indices:
            ts = th.tensor([i]  * x.shape[0], device=x.device)

            x.requires_grad_()   
            out = self.p_sample(model, x, ts) #self.ddim_sample(model, x, ts)

            # calculate the gradient
            diff = self.mask * (self.knowns - to_space(out['pred_xstart'])) # mask * (y - F * x_0)
            norm = th.linalg.norm(diff)  
            norm_grad = th.autograd.grad(outputs=norm, inputs=x)[0] # \nabla_x |y - F * x_0|^2
            # fix the gradient
            x = out['sample'] - norm_grad * 0.1  # x_t - grad * hyperparam

            x.detach_()
        return x

    def ddpm(self, model, x, t):
        ts = th.tensor([t]  * x.shape[0], device=x.device)
        return self.p_sample(model, x, ts)['sample']
    
    def ddim(self, model, x, t):
        ts = th.tensor([t]  * x.shape[0], device=x.device)
        return self.ddim_sample(model, x, ts)['sample']

    @th.no_grad()
    def _loop_real(self, model, denoise_fn, progress=False, device="cpu"):
        
        _indices = list(range(50))[::-1]
        # _indices = list(range(self.num_timesteps))[::-1]
        indices = tqdm(_indices) if progress else _indices
        Ts = th.tensor([_indices[0]]  * self.knowns.shape[0], device=self.knowns.device)
        x = self.q_sample(from_space(self.knowns).real, Ts) 
        # x = th.randn_like(self.knowns.real, device=device)
        for i in indices:  
            x = denoise_fn(model, x, i) 

        return x