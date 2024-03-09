import pdb
import shutil
from share import *

import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import math
F = torch.nn.functional

import time
import yaml
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def interpolate_linear(p0,p1, frac):
    return p0 + (p1 - p0) * frac

@torch.no_grad()
def slerp(p0, p1, fract_mixing: float):
    r""" Copied from lunarring/latentblending
    Helper function to correctly mix two random variables using spherical interpolation.
    The function will always cast up to float64 for sake of extra 4.
    Args:
        p0: 
            First tensor for interpolation
        p1: 
            Second tensor for interpolation
        fract_mixing: float 
            Mixing coefficient of interval [0, 1]. 
            0 will return in p0
            1 will return in p1
            0.x will return a mix between both preserving angular velocity.
    """ 
    if p0.dtype == torch.float16:
        recast_to = 'fp16'
    else:
        recast_to = 'fp32'
    
    p0 = p0.double()
    p1 = p1.double()
    norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
    epsilon = 1e-7
    dot = torch.sum(p0 * p1) / norm
    dot = dot.clamp(-1+epsilon, 1-epsilon)
    
    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * fract_mixing
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    interp = p0*s0 + p1*s1
    
    if recast_to == 'fp16':
        interp = interp.half()
    elif recast_to == 'fp32':
        interp = interp.float()
    return interp

    
class ContextManager:
    def __init__(self, version='2.1'):
        self.filters = {}
        self.mode = None
        self.version = version
        self.model = create_model('./controlnet/models/cldm_v21.yaml').cuda()
        self.ddim_sampler = DDIMSampler(self.model)
        self.model.load_state_dict(load_state_dict('./controlnet/models/openpose-sd21.ckpt', location='cuda'))


    def interpolate_new(self, img1, img2,  scale_control=1.5, prompt=None, n_prompt=None, min_steps=.3, max_steps=.55, ddim_steps=250,  guide_scale=7.5,  optimize_cond=0,  cond_lr=1e-4, bias=0, ddim_eta=0, out_dir='blend'):
        torch.manual_seed(49)
        if min_steps < 1:
            min_steps = int(ddim_steps * min_steps)
        if max_steps < 1:
            max_steps = int(ddim_steps * max_steps)
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
        
        if isinstance(img1, Image.Image):
            img1.save(f'{out_dir}/{0:03d}.png')
            img2.save(f'{out_dir}/{2:03d}.png')
            if img1.mode == 'RGBA':#
                    img1 = img1.convert('RGB')
            if img2.mode == 'RGBA':
                img2 = img2.convert('RGB')
            img1 = torch.tensor(np.array(img1)).permute(2,0,1).unsqueeze(0).cuda()
            img2 = torch.tensor(np.array(img2)).permute(2,0,1).unsqueeze(0).cuda()
        
        ldm = self.model
        ldm.control_scales = [1] * 13

        cond1 = ldm.get_learned_conditioning([prompt])
        uncond_base = ldm.get_learned_conditioning([n_prompt])
        cond = {"c_crossattn": [cond1], 'c_concat': None}
        un_cond = {"c_crossattn": [uncond_base], 'c_concat': None}
        
        self.ddim_sampler.make_schedule(ddim_steps, ddim_eta=ddim_eta, verbose=False)#构造ddim_timesteps,赋值给timesteps
        timesteps = self.ddim_sampler.ddim_timesteps

        left_image= ldm.get_first_stage_encoding(ldm.encode_first_stage(img1.float() / 127.5 - 1.0))
        right_image= ldm.get_first_stage_encoding(ldm.encode_first_stage(img2.float() / 127.5 - 1.0))

        
        kwargs = dict(cond_lr=cond_lr, cond_steps=optimize_cond, prompt=prompt, n_prompt=n_prompt, ddim_steps=ddim_steps, guide_scale=guide_scale, bias=bias, ddim_eta=ddim_eta, scale_control=scale_control)
        yaml.dump(kwargs, open(f'{out_dir}/args.yaml', 'w'))
        
        cur_step=140
        t = timesteps[cur_step]
        
        
        interpolation_type="noisediffusion"
        
        if interpolation_type!= 1:
            l1, _ = self.ddim_sampler.encode(left_image, cond, cur_step, 
            use_original_steps=False, return_intermediates=None,
            unconditional_guidance_scale=1, unconditional_conditioning=un_cond)
            l2, _ = self.ddim_sampler.encode(right_image, cond, cur_step, 
            use_original_steps=False, return_intermediates=None,
            unconditional_guidance_scale=1, unconditional_conditioning=un_cond)            
                    
        for num in range(5):
            frac_list=[0.17,0.33,0.5,0.67,0.83]
            name_list=[1,3,5,7,9]
            frac=frac_list[num]
            name=name_list[num]
            latent_frac=frac
            noise = torch.randn_like(left_image)
            if interpolation_type=="noise":
                l1 = ldm.sqrt_alphas_cumprod[t] * left_image + ldm.sqrt_one_minus_alphas_cumprod[t] * noise
                l2 = ldm.sqrt_alphas_cumprod[t] * right_image + ldm.sqrt_one_minus_alphas_cumprod[t] * noise
                noisy_latent = slerp(l1, l2, latent_frac)
                
            if interpolation_type=="slerp":       
                noisy_latent = slerp(l1, l2, latent_frac)

            if interpolation_type=="noisediffusion":
                coef=2.0
                gamma=0
                alpha=math.cos(math.radians(latent_frac*90))
                beta=math.sin(math.radians(latent_frac*90))
                l=alpha/beta
                
                alpha=((1-gamma*gamma)*l*l/(l*l+1))**0.5
                beta=((1-gamma*gamma)/(l*l+1))**0.5
    
                mu=1.2*alpha/(alpha+beta)
                nu=1.2*beta/(alpha+beta)
                
                l1=torch.clip(l1,-coef,coef)  
                l2=torch.clip(l2,-coef,coef)   
                
                noisy_latent= alpha*l1+beta*l2+(mu-alpha)*ldm.sqrt_alphas_cumprod[t] * left_image+(nu-beta)*ldm.sqrt_alphas_cumprod[t] * right_image+gamma*noise*ldm.sqrt_one_minus_alphas_cumprod[t]
                
                noisy_latent=torch.clip(noisy_latent,-coef,coef)    
            
                               
            samples= self.ddim_sampler.decode(noisy_latent, cond, cur_step, # cur_step-1 / new_step-1
                unconditional_guidance_scale=guide_scale, unconditional_conditioning=un_cond,
                use_original_steps=False)                
                            
            image = ldm.decode_first_stage(samples)

            image = (image.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            Image.fromarray(image[0]).save(f'{out_dir}/{name:03d}.png')

