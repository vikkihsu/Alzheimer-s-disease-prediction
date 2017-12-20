import nibabel as nib
from nilearn.plotting import plot_stat_map, show
import numpy as np
import cv2
import os
import collections

def getDescriptor(weight, sigma, step_size, mask):
    _, _, _, d = weight.shape
    for i in range(d):
        weight[:,:,:,i] = cv2.GaussianBlur(weight[:,:,:,i], (3,3), sigma)
    index = np.where(weight[...,0]!=0)
    
    x,y,z = np.meshgrid(range(min(index[1])-2,max(index[1])+3,step_size),
            range(min(index[0])-2,max(index[0])+3,step_size),
            range(min(index[2])-2,max(index[2])+3,step_size))
    smoothed_ROI = weight[x,y,z]
    
    index = np.where(smoothed_ROI[...,0]!=0)
    return smoothed_ROI[index].flatten()

#########################################################################
wgt_directory = "C:/Users/vikki/Desktop/MCI/weight_s"
#wgt_directory = "C:/Users/vikki/Desktop/MCI/weight_p"

i = 0; des = []
for root, _, files in os.walk(wgt_directory):
    for wgt_file in files:
        wgt = nib.load(os.path.join(root, wgt_file))
        w = wgt.get_data()
        des.append(getDescriptor(w,1,2))
        i += 1
        print('jiji:', i)
    break

with open('des_sMCI.npy', 'wb') as file:
    np.save(file, des)
'''
with open('des_pMCI.npy', 'wb') as file:
    np.save(file, des)
'''