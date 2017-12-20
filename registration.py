import pyelastix
import nibabel as nib
from nilearn.plotting import plot_stat_map, show
import numpy as np
import os

nii_directory = "C:/Users/vikki/Desktop/MCI/nii_s"
#nii_directory = "C:/Users/vikki/Desktop/MCI/nii_p"
followup = nib.load("followup.nii")
follow_up = np.ascontiguousarray(followup.get_data())
header = followup.header

images = []; nii_files = []
for root, _, files in os.walk(nii_directory):
    for nii_file in files:
        baseline = nib.load(os.path.join(root, nii_file))

        image = np.ascontiguousarray(baseline.get_data())
        image = (image)/np.std(image)
        images.append(image)
        nii_files.append(nii_file)
    break

params = pyelastix.get_default_params(type='BSPLINE')
params.NumberOfResolutions = 3
params.MaximumNumberOfIterations = 500
i = 0
for img, nii_file in zip(images, nii_files):
    imgg, field = pyelastix.register(img, follow_up, params, verbose=0)
    imgg = nib.Nifti1Image(imgg, followup.affine, header=header)
    nib.save(imgg, os.path.join(nii_file))
    i += 1
    print('jiji:', i)
#im3 = nib.load("test.nii.gz").get_data()
#print(im3[im3!=0])
    

