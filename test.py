#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:23:34 2020
"""
import torch
import torchvision.transforms as transforms
from networks import Generator_nu
from PIL import Image
import os
import numpy as np
import sys

# img_path = 'data/img/'
# save_path = 'data/pred/'
img_path = sys.argv[1]
save_path = sys.argv[2]

net_g = Generator_nu()
net_g.load_state_dict(torch.load('latest_G.pth'))
net_g.eval()
net_g.cuda()

transform_list = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

img_list = os.listdir(img_path)
for i in range(len(img_list)):
    img_name = img_list[i]
    Img = Image.open(img_path+img_name).convert('RGB')
    Input = transform_list(Img).unsqueeze(0).cuda()
    with torch.no_grad():
         pred = net_g(Input)
    out_img = pred.detach().squeeze(0).cpu().float().numpy()
    image_numpy = (np.transpose(out_img, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    result = image_numpy.copy()
    save_Img = Image.fromarray(result)
    save_Img.save(save_path+img_name)
    print('{}/{} done'.format(i+1,len(img_list)))
    
    
    
    
    
    
    
    
