import sys
from PIL import Image
import os, pickle
import pdb
osp = os.path
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
sys.path.append(osp.expandvars('$NFS/NoiseDiffusion/controlnet'))

import cm
CM = cm.ContextManager()
img1 = Image.open('controlnet/sample_imgs/bedroom1.png').resize((768, 768))
img2 = Image.open('controlnet/sample_imgs/bedroom2.png').resize((768, 768))

prompt='a photo of bed'
n_prompt='text, signature, logo, distorted, ugly, weird eyes, lowres, messy, weird face, lopsided, disfigured, bad art, poorly drawn, low quality, drawing, blurry, faded'
CM.interpolate_new(img1, img2,  prompt=prompt, n_prompt=n_prompt, ddim_steps=200,  guide_scale=10,  out_dir='controlnet/sample_results/bedroom')
