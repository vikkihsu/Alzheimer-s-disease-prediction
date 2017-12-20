import pyelastix
import nibabel as nib
from nilearn.plotting import plot_stat_map, show
import numpy as np
import os
import get_weight_image

reg_directory = "C:/Users/vikki/Desktop/MCI/reg_s"
#reg_directory = "C:/Users/vikki/Desktop/MCI/reg_p"

images = []; header = []; aff = []
for root, _, files in os.walk(reg_directory):
    for reg_file in files:
        img = nib.load(os.path.join(root, reg_file))
        image = img.get_data()

        aff.append(img.affine)
        images.append(image)
        header.append(img.header)
    break

mask_left = nib.load("mask_left.nii")
mask_right = nib.load("mask_right.nii")
L = get_weight_image.getdata(mask_left)
R = get_weight_image.getdata(mask_right)

j = 0
for i in np.arange(0,len(images),2):
    weight = get_weight_image.getWeight(images[i], images[i+1], R, 10e-9) + get_weight_image.getWeight(images[i], images[i+1], L, 10e-9)
    
    weight_both = nib.Nifti1Image(weight, aff[i], header=header[i])
    nib.save(weight_both, os.path.join('weight'+str(i)+'.nii.gz'))
    j += 1
    print('jiji:',j)
