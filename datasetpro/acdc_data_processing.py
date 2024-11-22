import h5py
import glob
import os
import numpy as np
import SimpleITK as sitk

slice_num = 0
mask_path = sorted(glob.glob("../data/image/*.nii.gz"))

for case in mask_path:
    print(case)
    img_itk = sitk.ReadImage(case)
    origin = img_itk.GetOrigin()
    spacing = img_itk.GetSpacing()
    direction = img_itk.GetDirection()
    image = sitk.GetArrayFromImage(img_itk)
    msk_path = case.replace("image", "label").replace(".nii.gz", "_gt.nii.gz")

    if mask_path:
        mask_itk = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(img_itk)
        bg_label = 1
        left_ventricle_label = 0.75
        myocardium_label = 0.5
        right_ventricle_label = 0.25
        mask_left = np.where(mask == left_ventricle_label, 3, 0)
        mask_right = np.where(mask == right_ventricle_label, 1, 0)
        mask_bg = np.where(mask == bg_label, 4, 0)
        mask_myo = np.where(mask == myocardium_label, 2, 0)
        mask = mask_left + mask_right + mask_bg + mask_myo

        mask = np.where(mask != 0, 1, 0)


        image = (image - image.min()) / (image.max() - image.min())
        image = image.astype(np.float32)

        item = case.split("/")[-1].split(".")[0]
        if image.shape != mask.shape:
            print("Error")
        f = h5py.File(
            '../data/ACDC_RLV/data/{}.h5'.format(item), 'w')
        f.create_dataset(
            'image', data=image, compression="gzip")
        f.create_dataset('label', data=mask, compression="gzip")
        f.close()
        #
        for slice_ind in range(image.shape[0]):
            f = h5py.File(
                '../data/ACDC_RLV/data/slices/{}_slice_{}.h5'.format(item, slice_ind), 'w')
            f.create_dataset(
                'image', data=image[slice_ind], compression="gzip")
            f.create_dataset('label', data=mask[slice_ind], compression="gzip")
            f.close()
            slice_num += 1
