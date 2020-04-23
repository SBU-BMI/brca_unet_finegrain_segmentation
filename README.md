# brca_unet_finegrain_segmentation

This repo is for training and running prediction of brca finegrain segmentation using Unet model.


## Dependencies

 - [Pytorch 0.4.0](http://pytorch.org/)
 - Torchvision 0.2.0
 - cv2 (3.4.1)
 - [Openslide 1.1.1](https://openslide.org/api/python/)
 - [sklearn](https://scikit-learn.org/stable/)
 - [PIL](https://pillow.readthedocs.io/en/3.1.x/reference/Image.html)

## Setup data folder

- The training/validation data are saved in the .data folder
- Training patches and corresponding masks are stored in separate folders. Here is an example:
- Training data:
    + TCGA_BRCA_finegrain_patches_10X
    + TCGA_BRCA_finegrain_patches_10X_mask
- Validation data:
    + TCGA_BRCA_finegrain_patches_10X_val
    + TCGA_BRCA_finegrain_patches_10X_mask_val
- Number 10X indicates the resolution in which the patches were extracted. There is an argument in train.py, 'resolution', to select the resolution if you have different resolution settings.
- The training patch and the corresponding mask have the same size, ideally square patch, e.g 1000x1000 pixels.
- The mask contain values 0, 1, 2,... to indicate the class ID. For binary classification, mask's values are 0/1
- Change the data folder paths accordingly in util_codes/utils.py to load training/validation data

## Training

- Run python train.py to train the model.
- Arguments can be passed to select input size of the model, batch size, learning rate, etc.

## Prediction on testset

- Run python predict_testset.py to run prediction for testing patches.
- Predicted patches will be stored in data folder, under the name "predicted_imgs_10X", if the patches are at 10X
- The predicted patches are original patches overlayed with predicted masks.

## Prediction on WSIs

- Either run python prediction_WSIs_single_machine.py or python prediction_WSIs_multiple_machines.py to run prediction on WSIs
- Use prediction_WSIs_single_machine.py if you have only 1 machine to use
- Use prediction_WSIs_multiple_machines.py on different machine to process multiple WSIs at the same time
- Arguments to be changed in the __main__ function including wsi_fol, wsi_extension, model_path, out_fol
- All WSIs in the wsi_fol with extension wsi_extension will be processed. The predicted masks are saved in corresponding folder in out_fol

## Generaate polygon jsons for predicted masks

- Run python generate_jsons.py to generate jsons for WSIs
- Arguments to be changed in the __main__ function including wsi_fol and out_fol
- The output for each WSI is one json file stored in the folder that contains its predicted masks
