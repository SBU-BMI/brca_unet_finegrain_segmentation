import torch

from unet import UNet
from util_codes.utils import *
from train import get_data_transforms
from torch.autograd import Variable


class predict_WSI:
    def __init__(self, model_path, no_classes=2):
        self.model_path = model_path
        self.no_classes = no_classes
        self.data_transforms = get_data_transforms()
        self.net = self.load_model()

    def load_model(self):
        print('Loading model from ', self.model_path)
        try:
            net = UNet(n_channels=3, n_classes=self.no_classes, bilinear=False)
        except:
            net = UNet(n_channels=3, n_classes=self.no_classes, bilinear=True)

        net = parallelize_model(net)
        net.load_state_dict(torch.load(self.model_path))
        print("Model loaded!")
        net.eval()
        return net

    def predict_large_patch(self, img):
        device = torch.device("cuda:0")

        img = Image.fromarray(img.astype(np.uint8), 'RGB')
        img = self.data_transforms['val'](img)
        img.unsqueeze_(0)
        img = Variable(img.to(device))

        masks_pred = self.net(img)
        _, masks_pred = torch.max(masks_pred.data, 1)
        masks_pred.squeeze_(0)
        masks_pred = masks_pred.data.cpu().numpy()

        return masks_pred


