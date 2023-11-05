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
        # x_real.requires_grad_()   
        # x_0 = self.p_mean_variance(model, x_real, ts)["pred_xstart"] # predictor
        # x_0 = self.projector(x_0, t)
        # alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, ts, x_real.shape)
        # x_real_pre = th.sqrt(alpha_bar_prev) * x_0  + th.sqrt(1-alpha_bar_prev) * th.rand_like(x_0)

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
    def ppn_complex(self, model, x_complex, t):
        # x_complex=get_mvue(to_space(x_complex), self.sens)
        # x_complex = x_complex * self.sens
        
        x_real = from_mc(x_complex)
        if t > -1:
            x_real = self.ppn(model, x_real, t)
        else:
            x_real = self.dps(model, x_real, t)
        x_complex = to_mc(x_real, self.coilNum)
    
        # x_complex = x_complex * th.conj(self.sens)
        # combine condition
        # if self.mixstepsize > 0 and t % self.mixstepsize == 0:
        #     lamb = 1. #lamb_schedule.get_current_lambda(t)
            # x = to_mc(x)
            # # TODO
            # knowns = to_mc(self.knowns)
            # x = self.kaczmarz(x, knowns, lamb=lamb)
            # x = from_mc(x)
            
        return x_complex


    def A(self, x):
        return self.mask * to_space(self.sens * x)

    def A_H(self, x):  # Hermitian transpose
        return th.sum(th.conj(self.sens) * from_space(x * self.mask), dim=1).unsqueeze(dim=1)

    def kaczmarz(self, x, y,lamb=1.0): #[1, 15, 320, 320])
        x = x + lamb * self.A_H(y - self.A(x)) # [1, 15, 320, 320]) + [1, 1, 320, 320])
        return x

    def noisor(self, x_0_hat, t, x_0): # alg 1
        # recip_snr = _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_0.shape)
        # ss = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        # x_0_hat += th.randn_like(x_0)*recip_snr**2  # noise version  
        # s1 = (x_0_hat-x_0)/recip_snr 
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x_0.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x_0.shape)
        mean = th.sqrt(alpha_bar_prev) * x_0_hat + th.sqrt(alpha_bar*(1-alpha_bar_prev)/(1-alpha_bar)) * (x_0_hat-x_0)
        noise = th.sqrt((1-alpha_bar_prev)*(1-alpha_bar)) * th.randn_like(x_0)
        return mean + noise

    # def noisor(self, x_0_hat, t, x_0): # alg 1
    #     # recip_snr = _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_0.shape)
    #     # ss = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    #     # x_0_hat += th.randn_like(x_0)*recip_snr**2  # noise version  
    #     # s1 = (x_0_hat-x_0)/recip_snr 
        
    #     alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x_0.shape)
    #     alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x_0.shape)
    #     if (1-self.alphas_cumprod[t[0]]) < 0.6:
    #         mean = th.sqrt(alpha_bar_prev)  * x_0 #(x_0_hat + x_0)/2
    #         noise = th.sqrt(1-alpha_bar_prev - 0.0 *alpha_bar_prev *(1-alpha_bar)) * th.randn_like(x_0)
    #     else: 
    #         mean = th.sqrt(alpha_bar_prev) * x_0 + th.sqrt(alpha_bar*(1-alpha_bar_prev)/(1-alpha_bar)) *  (x_0_hat-x_0)
    #         noise = th.sqrt((1-alpha_bar_prev)*(1-alpha_bar)) * th.randn_like(x_0)
    #     return mean + noise


    # def noisor(self, x_0_hat, t, x_0): # alg 1
    #     alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x_0.shape)
    #     alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x_0.shape)
    #     mean = th.sqrt(alpha_bar_prev) * x_0 + th.sqrt(alpha_bar*(1-alpha_bar_prev)/(1-alpha_bar)) *  (x_0_hat-x_0)
    #     noise = th.sqrt((1-alpha_bar_prev)*(1-alpha_bar)) * th.randn_like(x_0)
    #     return mean + noise
    
    

    # def ppn(self, model, x_real, t): # 0: x_0, 1: x_t
    #     ts = th.tensor([t]  * x_real.shape[0], device=x_real.device)
    #     x_0 = self.p_mean_variance(model, x_real, ts)["pred_xstart"] # predictor
    #     x_0_hat = self.projector(x_0)

    #     # alpha_bar = _extract_into_tensor(self.alphas_cumprod, ts, x_0.shape)
        
    #     # x_space = to_space(x_0)
    #     # x_space = merge_known_with_mask(x_space, self.knowns, self.mask)
    #     # # print(alpha_bar)
    #     # x_0_hat = from_space(x_space).real
        
    #     xx_0 = self.noisor(x_0_hat, ts, x_0)
    #     return xx_0


    # def ppn(self, model, x_real, t): # 0: x_0, 1: x_t
    #     ts = th.tensor([t]  * x_real.shape[0], device=x_real.device)
    #     x_0 = self.p_mean_variance(model, x_real, ts)["pred_xstart"] # predictor
        
    #     x_0_hat = self.projector(x_0)
    #     # if self.mc_projector and (t+1) % 10 == 0:
    #     #     # print(t, t/50)
    #     #     x_0_hat =  self.mc_projector(x_0_hat, (t+1)/200)

    #     x_0 = self.noisor(x_0_hat, ts, x_0)

    #     # if t ==0:   
    #     #     x_0 = apply_sparse(x_0, 0.002)
    #         # x_0 = apply_sparse(x_0, 0.2)
        
    #     return x_0
    
    @th.no_grad()
    def ppn(self, model, x_real, t): # 0: x_0, 1: x_t
        ts = th.tensor([t]  * x_real.shape[0], device=x_real.device)
        x_0 = self.p_mean_variance(model, x_real, ts)["pred_xstart"] # predictor
        # save_img(to_mc(x_0, self.coilNum)[2].abs().squeeze().detach().cpu(), "img_%02d.jpg"%t)
        # save_img(to_mc(x_0, self.coilNum)[2].abs().squeeze().detach().cpu(), "_img_%02d.jpg"%t)
        x_0_hat = self.projector(x_0, t)
        
        # alpha_bar = _extract_into_tensor(self.alphas_cumprod, ts, x_real.shape)
        # for i in range(0):
        #     x_real = th.sqrt(alpha_bar) * x_0  + th.sqrt(1-alpha_bar) * th.rand_like(x_real)
        #     x_0 = self.p_mean_variance(model, x_real, ts)["pred_xstart"] # predictor

        # if self.mc_projector and (t+1) % 10 == 0:
        #     x_0 =  self.mc_projector(x_0, (t+1)/50)


        # ppn
        # alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, ts, x_real.shape)
        # x_real_pre = th.sqrt(alpha_bar_prev) * x_0_hat  + th.sqrt(1-alpha_bar_prev) * th.rand_like(x_0_hat)

        x_real_pre = self.noisor(x_0_hat, ts, x_0)

        
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
            if isComplex:

                # x_comb method
                def projector_complex(x_real,t=0): #[2, 1, 320, 320])
                    x_complex = to_mc(x_real, self.coilNum)  #[1, 1, 320, 320])
                    x_coils = from_mvue(x_complex, self.sens) #[1, 20, 320, 320])
                    # x_coils = self.sens * x_complex  #[1, 20, 320, 320])
                    x_space = to_space(x_coils)

                    # x_space = th.conj(self.sens) * (self.knowns * self.mask +  x_space * (1. - self.mask))
                    x_space =  (self.knowns * self.mask +  x_space * (1. - self.mask))
                    
                    # x_space = merge_known_with_mask(x_space, self.knowns, self.mask)
                    return from_mc(to_mvue(from_space(x_space), self.sens, self.mcType=='a'))

                # def projector_complex(x_real, t=0):
                #     x_complex = to_mc(x_real, self.coilNum)  # [15, 1, 320, 320])
                #     x_space = to_space(x_complex)
                #     x_space = merge_known_with_mask(x_space, self.knowns, self.mask, coeff=1)
                #     return from_mc(from_space(x_space))
                
     
                def projector_complex2(x_real, lmd):
                    x_complex = to_mc(x_real, self.coilNum)  # [1, 15, 320, 320])
                    # x_complex=rearrange(x_complex, '(b c) 1 h w -> b c h w', c=15)
                    
                    x_proj =  self.mask * ( self.knowns  - self.mask * to_space(self.sens * x_complex))
                    x = th.conj(self.sens)*from_space(x_proj) #[1, 20, 320, 320])
                    diff =  th.sum(x,dim=1, keepdim=True) #([1, 1, 320, 320])
                    return from_mc(x_complex + lmd * diff)

                self.projector = projector_complex
                self.mc_projector = projector_complex2
                sample_fn = partial(self._loop_complex, model=model, progress=progress, device=device)
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

    # def noisor(self, x_0_hat, t, x_0): # alg 1
    #     recip_snr = _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_0.shape)
    #     ss = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    #     # x_0_hat += th.randn_like(x_0)*recip_snr**2  # noise version  
    #     # s1 = (x_0_hat-x_0)/recip_snr 

    #     s1 = (x_0_hat-x_0)/recip_snr + th.randn_like(x_0)*recip_snr

    #     alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x_0.shape)
    #     x = th.sqrt(alpha_bar_prev) * x_0_hat + th.sqrt(1-alpha_bar_prev)* s1
    #     return x
    


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

        # nn = 99
        # _indices = list(range(nn))[::-1]
        # indices = tqdm(_indices) if progress else _indices
        # ts = th.tensor([nn]  * x.shape[0], device=x.device)
        # x = self.q_sample(x, ts)
        # for i in indices:
        #     x = self.p_sample(model, x, th.tensor([i]  * x.shape[0], device=x.device))['sample']
        
        return x
    
    # @th.no_grad()
    def _loop_complex(self, model, progress=False, device="cpu"):
        class lambda_schedule_linear():
            def __init__(self, start_lamb=1.0, end_lamb=0.2):
                super().__init__()
                self.start_lamb = start_lamb
                self.end_lamb = end_lamb

            def get_current_lambda(self, i):
                return self.start_lamb + (self.end_lamb - self.start_lamb) * (i / 100)
        self.lamb_schedule = lambda_schedule_linear()

        _indices = list(range(50))[::-1]
        # _indices = list(range(self.num_timesteps))[::-1]
        indices = tqdm(_indices) if progress else _indices
        # x_complex = th.randn_like(self.knowns[:,:1]) if self.mcType=='a' else th.randn_like(self.knowns) #[1, 1, 320, 320])
        x_complex = th.randn_like(self.knowns[:,:1]) if self.mcType=='a' else th.randn_like(self.knowns)
        for i in indices:  
            x_complex = self.ppn_complex(model, x_complex, i) 

        # x_complex=rearrange(x_complex, 'b c h w -> (b c) 1 h w', c=20)
        # x_complex = try_rss_complex(x_complex)
        # b = rss_complex2(x_complex)
        # rr = np.allclose(a.cpu(),b.cpu())
        # x_complex/= x_complex.abs().max()
        return x_complex