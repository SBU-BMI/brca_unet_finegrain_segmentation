import os
from util_codes import patch_extraction, data_loader_WSI, predict_WSI
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
import glob
import time

def mkdir(fol):
    if not os.path.exists(fol):
        os.mkdir(fol)

def create_done_file(out_fol_wsi):
    done_file = os.path.join(out_fol_wsi, 'prediction_done.txt')
    os.system(f'touch {done_file}')

class run_prediction_WSIs:
    def __init__(self, wsi_fol, wsi_extension, out_fol, model_path):
        self.wsi_fol = wsi_fol
        self.wsi_extension = wsi_extension
        self.wsi_paths = self.get_wsi_paths()
        self.out_fols = {wsi:os.path.join(out_fol, self.get_wsi_id(wsi)) for wsi in self.wsi_paths}
        self.predict_WSI_handler = predict_WSI(model_path, no_classes=2)

    def get_wsi_paths(self):
        paths = glob.glob(os.path.join(self.wsi_fol, '*.' + self.wsi_extension))
        return paths

    def get_wsi_id(self, wsi_path):
        return wsi_path.rstrip('/').split('/')[-1]

    def run_prediction(self, data_loader, out_fol_wsi):
        for index, data in enumerate(data_loader):
            patches, fnames = data
            predicted_masks = self.predict_WSI_handler.predict_large_patch(patches)

            print(f"Predicting: {index + 1}/{len(data_loader)}")

            for i, fname in enumerate(fnames):
                fname_path = os.path.join(out_fol_wsi, fname)
                predicted_mask = predicted_masks[i] * 255
                cv2.imwrite(fname_path, predicted_mask.astype(np.uint8))

    def run_prediction_one_WSI(self, wsi_path):
        out_fol_wsi = self.out_fols[wsi_path]
        mkdir(out_fol_wsi)
        print('Start predicting: ', wsi_path)
        start = time.time()
        patch_extraction_handler = patch_extraction(wsi_path, patch_size_10X=1000)

        imgs_set_complete = data_loader_WSI(patch_extraction_handler, isComplete=True)
        imgs_set_partial = data_loader_WSI(patch_extraction_handler, isComplete=False)

        data_loader_complete = DataLoader(imgs_set_complete, batch_size=4, shuffle=False, num_workers=16)
        data_loader_partial = DataLoader(imgs_set_partial, batch_size=1, shuffle=False, num_workers=16)

        self.run_prediction(data_loader_complete, out_fol_wsi)
        self.run_prediction(data_loader_partial, out_fol_wsi)

        create_done_file(out_fol_wsi)
        print('Done predicting: {}. Time: {:.2f}mins'.format(wsi_path, (time.time() - start)/60))

    def main(self):
        for wsi_path in self.wsi_paths:
            self.run_prediction_one_WSI(wsi_path)


def main(wsi_fol, wsi_extension, out_fol, model_path):
    run_prediction_WSIs_handler = run_prediction_WSIs(wsi_fol, wsi_extension, out_fol, model_path)
    run_prediction_WSIs_handler.main()

if __name__ == '__main__':
    wsi_fol = '/data01/shared/hanle/svs_tcga_seer_brca'
    wsi_extension = 'svs'
    model_path = '.model/CP1860_resolution10_APS448_Test_upLearned_best_0.8005.pth'
    out_fol = '.'
    mkdir(out_fol)

    main(wsi_fol, wsi_extension, out_fol, model_path)







