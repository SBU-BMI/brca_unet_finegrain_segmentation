import os, sys
from patch_extraction import patch_extraction
from predict_WSI import predict_WSI
import cv2
import time


def mkdir(fol):
    if not os.path.exists(fol):
        os.mkdir(fol)


if __name__ == '__main__':
    wsi_path = '/data01/shared/hanle/svs_tcga_seer_brca/TCGA-E2-A153-01Z-00-DX1.svs'
    model_path = '/data02/shared/hanle/brca_unet_finegrain_segmentation/checkpoints/CP1860_resolution10_APS448_Test_upLearned_best_0.8005.pth'
    out_fol = wsi_path.split('/')[-1]
    mkdir(out_fol)

    patch_extraction_handler = patch_extraction(wsi_path, patch_size_10X=1000)
    predict_WSI_handler = predict_WSI(model_path, no_classes=2, APS=250)
    start = time.time()
    len_coors = len(patch_extraction_handler.coors)

    while patch_extraction_handler.has_next():
        patch, fname = patch_extraction_handler.next_patch()
        fname_path = os.path.join(out_fol, fname)
        if patch is not None:
            time_elapsed = (time.time() - start)/60

            print("Predicting patch {} - {}: {}/{} \t time_elapsed: {} \t time_remaining: {}".
                  format(fname,
                        patch.shape,
                        patch_extraction_handler.index,
                        len_coors,
                        time_elapsed,
                        time_elapsed*len_coors/patch_extraction_handler.index))

            predicted_mask = predict_WSI_handler.predict_large_patch(patch)
            cv2.imwrite(fname_path, predicted_mask)


