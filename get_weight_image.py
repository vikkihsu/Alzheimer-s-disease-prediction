import nibabel as nib
from nilearn.plotting import plot_stat_map, show
import numpy as np
import os
import collections

def getdata(image):
    #image should be nib.load(".nii")
    #plot_stat_map(image)
    #show()
    #transform the nii to matrix-like image
    return image.get_data()

#the number of first dimesion in weight is the number of pixels in ROI
def getWeight(image_B, image_F, mask, epsilon):
    #get ROI and their index
    #get ROI from basline and followup
    #keep the pixel values which are in ROI
    ROI_B = np.where(mask==1, image_B, 0)
    ROI_F = np.where(mask==1, image_F, 0)

    ROI_index_B = np.where(ROI_B!=0)
    ROI_index_F = np.where(ROI_F!=0)
    
    ROI_B = ROI_B[min(ROI_index_B[0])-2:max(ROI_index_B[0])+3, 
        min(ROI_index_B[1])-2:max(ROI_index_B[1])+3, 
        min(ROI_index_B[2])-2:max(ROI_index_B[2])+3]
    ROI_F = ROI_F[min(ROI_index_F[0])-2:max(ROI_index_F[0])+3, 
        min(ROI_index_F[1])-2:max(ROI_index_F[1])+3, 
        min(ROI_index_F[2])-2:max(ROI_index_F[2])+3]

    h_shape, w_shape, d_shape = ROI_B.shape
    weight = np.zeros((h_shape,w_shape,d_shape,27))
    #the number of first dimesion in weight is the number of pixels in ROI
    new = np.zeros((3,3,3))
    h, w, d = np.where(ROI_B!=0)
    for i, j, t in zip(h,w,d):
        down = float('inf')
        for a in range(-1,2):
            for b in range(-1,2):
                for c in range(-1,2):
                    B = ROI_B[i-1:i+2,j-1:j+2,t-1:t+2]
                    F = ROI_F[i-1+a:i+2+a,j-1+b:j+2+b,t-1+c:t+2+c]
                    new[a+1,b+1,c+1] = np.sum(((B-np.sum(B)/np.std(B))-(F-np.sum(F))/np.std(F))**2)
                    if down > new[a+1,b+1,c+1]:
                        down = new[a+1,b+1,c+1]
        if down != 0:
            weight[i,j,t] = np.exp(-new/down+epsilon).reshape(-1)

        #normalize each point in weight, so that the sum of all dimeansions in each point is 1.0
        weight[i,j,t] = weight[i,j,t]/np.sum(weight[i,j,t])
   
    #pad the weight to original image size
    h1, w1, d1 = image_B.shape
    weight = np.pad(weight, ((min(ROI_index_B[0])-2,h1-max(ROI_index_B[0])-3),
                (min(ROI_index_B[1])-2,w1-max(ROI_index_B[1])-3),
                (min(ROI_index_B[2])-2,d1-max(ROI_index_B[2])-3), (0,0)), 'constant')

    return weight