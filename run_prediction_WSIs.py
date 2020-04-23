import os, sys
from patch_extraction import patch_extraction, data_loader_WSI
from predict_WSI import predict_WSI
import cv2
import numpy as np
import time
from torch.utils.data import DataLoader, Dataset


def mkdir(fol):
    if not os.path.exists(fol):
        os.mkdir(fol)


if __name__ == '__main__':
    wsi_path = '/data01/shared/hanle/svs_tcga_seer_brca/TCGA-E2-A153-01Z-00-DX1.svs'
    model_path = '/data02/shared/hanle/brca_unet_finegrain_segmentation/checkpoints/CP1860_resolution10_APS448_Test_upLearned_best_0.8005.pth'
    out_fol = wsi_path.split('/')[-1]
    mkdir(out_fol)

    patch_extraction_handler = patch_extraction(wsi_path, patch_size_10X=1000)
    predict_WSI_handler = predict_WSI(model_path, no_classes=2)
    len_coors = len(patch_extraction_handler.coors)
    start = time.time()

    imgs_set = data_loader_WSI(patch_extraction_handler)
    data_loader = DataLoader(imgs_set, batch_size=2, shuffle=False, num_workers=8)

    for i, data in enumerate(data_loader):
        patches, fnames = data
        predicted_masks = predict_WSI_handler.predict_large_patch(patches)

        for i, fname in enumerate(fnames):
            fname_path = os.path.join(out_fol, fname)
            predicted_mask = predicted_masks[i]*255
            cv2.imwrite(fname_path, predicted_mask.astype(np.uint8))

        time_elapsed = (time.time() - start)/60
        print("Predicting patchs: {}/{} \t time_elapsed: {:.2f}mins \t time_remaining: {:.2f}mins".
              format(patch_extraction_handler.index,
                     len_coors,
                     time_elapsed,
                     time_elapsed*len_coors/patch_extraction_handler.index - time_elapsed))




