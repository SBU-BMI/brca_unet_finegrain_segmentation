import torch

from unet import UNet
from util_codes.utils import *
from train import get_data_transforms
from torch.autograd import Variable


class predict_WSI:
    def __init__(self, model_path, no_classes=2, APS=500):
        self.model_path = model_path
        self.no_classes = no_classes
        self.APS = APS
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

    def predict_large_patch(self, data):
        device = torch.device("cuda:0")
        h, w = data.shape[:2]  # data.shape 1000x1000x3
        num_rows, num_cols = h // self.APS, w // self.APS
        imgs = torch.empty(num_rows * num_cols, 3, self.APS, self.APS)
        predicted_mask = np.zeros((h, w))
        ind = 0
        for r in range(0, h, self.APS):
            for c in range(0, w, self.APS):
                img = data[r: r + self.APS, c: c + self.APS, :]
                img = Image.fromarray(img.astype(np.uint8), 'RGB')
                imgs[ind] = self.data_transforms['val'](img)  # 3xAPSxAPS
                ind += 1

        imgs = Variable(imgs.to(device))
        masks_pred = self.net(imgs)
        _, masks_pred = torch.max(masks_pred.data, 1)
        masks_pred = masks_pred.data.cpu().numpy()

        ind = 0
        for r in range(0, h, self.APS):
            for c in range(0, w, self.APS):
                predicted_mask[r: r + self.APS, c: c + self.APS] = masks_pred[ind]
                ind += 1

        return predicted_mask


