# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 16:14:57 2020
"""
import numpy as np
CML = np.load('CML_feature.npy',allow_pickle=True).item()
Norm = np.load('Normal_feature.npy',allow_pickle=True).item()

CML_case_num = len(CML['case_ids'])
Norm_case_num = len(Norm['case_ids'])

# props 1~4:mean,std,max,min
feature_name_list = ['mk_num','wbc_num','mk_density','mk_radious1','mk_radious2','mk_radious3','mk_radious4']

feature_num = len(feature_name_list)
CML_feature_matrix = np.zeros((CML_case_num,feature_num))
Norm_feature_matrix = np.zeros((Norm_case_num,feature_num))

for ii in range(CML_case_num):
    CML_feature_matrix[ii,0] = CML['mk_num'][ii]
    CML_feature_matrix[ii,1] = CML['wbc_num'][ii]
    CML_feature_matrix[ii,2] = CML['mk_density'][ii]
    CML_feature_matrix[ii,3:7] = CML['mk_radious'][ii]

for jj in range(Norm_case_num):
    Norm_feature_matrix[jj,0] = Norm['mk_num'][jj]
    Norm_feature_matrix[jj,1] = Norm['wbc_num'][jj]
    Norm_feature_matrix[jj,2] = Norm['mk_density'][jj]
    Norm_feature_matrix[jj,3:7] = Norm['mk_radious'][jj]

## Just check statical feature (mean,std,max,min)
CML_sf = CML_feature_matrix[:,3:7]
Norm_sf = Norm_feature_matrix[:,3:7]

x1 = CML_sf[:,2]
x2 = CML_sf[:,3]
y1 = Norm_sf[:,2]
y2 = Norm_sf[:,3]
bar_width = 0.4
import matplotlib.pyplot as plt
x = np.arange(len(x1))
y = np.arange(len(y1))

plt.subplot(121)
plt.bar(x,x1,bar_width,color='blue',label='max')
plt.bar(x+bar_width,x2,bar_width,color='green',label='min')
plt.title('CML')
plt.legend()

plt.subplot(122)
plt.bar(y,y1,bar_width,color='blue',label='max')
plt.bar(y+bar_width,y2,bar_width,color='green',label='min')
plt.title('Norm')
plt.legend()














