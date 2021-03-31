#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:44:15 2019
"""
from skimage import io,morphology,measure
import os
import numpy as np
import time

def sort_case_index(img_path):
    img_list = os.listdir(img_path)
    case_list = []
    id_list = []
    for i in range(len(img_list)):
        img_name = img_list[i]
        a,b = img_name.split('_')
        case_list.append(a)
        id_list.append(b[:-4])
        
    case_ids = list(set(case_list))
    img_ids = []
    for j in range(len(case_ids)):
        case_name = case_ids[j]    
        case_index = [i for i,x in enumerate(case_list) if x==case_name]        
        index_ = []
        for jj in range(len(case_index)):
            img_index = id_list[case_index[jj]]           
            index_.append(img_index)
        img_ids.append(index_)
    return case_ids,img_ids    

def mask_post_process(image_numpy,L_T=4900,H_T=14400):  #16900
    MK_mask = (image_numpy[:,:,2] - image_numpy[:,:,0])>128
    MK_1 = morphology.remove_small_objects(MK_mask,L_T)   
    MK_2 = morphology.remove_small_objects(MK_1,H_T)    
    MK_ = (np.uint8(MK_1) - np.uint8(MK_2))*255
    
    b_mask = image_numpy[:,:,0]+np.uint8(MK_2)*255
    b_mask = b_mask>128
    b_mask = morphology.remove_small_holes(b_mask,900)
    b_mask = morphology.remove_small_objects(b_mask,900)
    b_mask = np.uint8(b_mask)*255
    
    MK_ = MK_>128
    MK_ = morphology.remove_small_holes(MK_,900)
    MK_ = morphology.remove_small_objects(MK_,900)
    MK_ = np.uint8(MK_)*255
    
    fmk_mask = np.zeros_like(image_numpy)
    fmk_mask[:,:,0] = b_mask
    fmk_mask[:,:,1] = b_mask
    fmk_mask[:,:,2] = b_mask + MK_
    
    return fmk_mask

def calculate_statical_features(mask):
    # Calculate MK's density
    cell_mask = mask[:,:,2]
    mk_mask = mask[:,:,2]-mask[:,:,0]
    
    cell_mask = morphology.erosion(cell_mask>0)
    labeled_cell_mask = morphology.label(cell_mask)
    
    labeled_mk_mask = morphology.label(mk_mask>0)
    
    cell_num = labeled_cell_mask.max()
    mk_num = labeled_mk_mask.max()
    
    mk_props = measure.regionprops(labeled_mk_mask)
    mk_radious = []
    for z in range(len(mk_props)):    
        radious = mk_props[z].equivalent_diameter/2
        mk_radious.append(radious)
    return cell_num,mk_num,mk_radious   

img_path = '../extracted_patches/batch1/Normal_Tiles/'
mask_path = '../extracted_patches/batch1/Normal_pred/'
       
case_ids,img_ids = sort_case_index(img_path) 
mk_num_list = []
all_wbc_num_list = []      
density_list = []
mk_radious_props_list = []

for i in range(len(case_ids)):    # patient
    mk = 0
    cell = 0
    mk_radious_list = []
    for j in range(len(img_ids[i])):   # patch
        s_time = time.time()
        img_name = case_ids[i]+'_'+img_ids[i][j]+'.png'
        img = io.imread(img_path+img_name)
        mask = io.imread(mask_path+img_name)
        mask = mask_post_process(mask,L_T=5000,H_T=25600)
        wbc_mask = mask[:,:,2]
        mk_mask = mask[:,:,2] - mask[:,:,0]
        
        # calculate statical features
        cell_num,mk_num,mk_radious = calculate_statical_features(mask)
        cell += cell_num
        mk += mk_num
        mk_radious_list += mk_radious
       
        e_time = time.time()
        print('{}/{},{}/{} done, taken {} S'.format(i+1,len(case_ids),j+1,len(img_ids[i]),e_time-s_time))
       
    mk_num_list.append(mk)
    all_wbc_num_list.append(cell)
    density_list.append(mk/cell)
    mk_radious_mean = np.mean(mk_radious_list)
    mk_radious_std = np.std(mk_radious_list)    
    mk_radious_max = np.max(mk_radious_list)
    mk_radious_min = np.min(mk_radious_list)
    mk_radious_props_list.append([mk_radious_mean,mk_radious_std,mk_radious_max,mk_radious_min])

# Feature sort
fdata = {}
fdata['case_ids'] = case_ids
fdata['mk_num'] = mk_num_list
fdata['wbc_num'] = all_wbc_num_list
fdata['mk_density'] = density_list
fdata['mk_radious'] = mk_radious_props_list
np.save('Normal_feature.npy',fdata)    

        
        
        
        





        
        
        
 














       
