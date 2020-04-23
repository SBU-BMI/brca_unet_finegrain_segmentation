import numpy as np
import openslide
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from train import get_data_transforms


class patch_extraction:
    def __init__(self, slide_name, patch_size_10X=1000):
        self.slide_name = slide_name
        self.patch_size_10X = patch_size_10X
        self.oslide, self.pw, self.width, self.height = self.get_oslide()
        self.coors, self.coors_partial = self.get_coors()
        self.index = 0

    def get_oslide(self):
        try:
            oslide = openslide.OpenSlide(self.slide_name)
            if openslide.PROPERTY_NAME_MPP_X in oslide.properties:
                mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X])
            elif "XResolution" in oslide.properties:
                mag = 10.0 / float(oslide.properties["XResolution"])
            elif 'tiff.XResolution' in oslide.properties:  # for Multiplex IHC WSIs, .tiff images
                Xres = float(oslide.properties["tiff.XResolution"])
                if Xres < 10:
                    mag = 10.0 / Xres
                else:
                    mag = 10.0 / (10000 / Xres)  # SEER PRAD
            else:
                print('[WARNING] mpp value not found. Assuming it is 40X with mpp=0.254!', self.slide_name)
                mag = 10.0 / float(0.254)

            pw = int(self.patch_size_10X * mag / 10)
            width = oslide.dimensions[0]
            height = oslide.dimensions[1]
            return oslide, pw, width, height
        except:
            print('{}: exception caught'.format(self.slide_name))
            exit(1)

    def get_coors(self):
        coors = []
        coors_partial = []
        for x in range(1, self.width, self.pw):
            for y in range(1, self.height, self.pw):
                if x + self.pw > self.width:
                    pw_x = self.width - x
                else:
                    pw_x = self.pw

                if y + self.pw > self.height:
                    pw_y = self.height - y
                else:
                    pw_y = self.pw

                if int(self.patch_size_10X * pw_x / self.pw) > 50 and int(self.patch_size_10X * pw_y / self.pw) > 50:
                    if pw_x == self.pw and pw_y == self.pw:
                        coors.append((x, y, pw_x, pw_y))
                    else:
                        coors_partial.append((x, y, pw_x, pw_y))

        return coors, coors_partial

    def extract_patch(self, corr):
        x, y, pw_x, pw_y = corr

        fname = '{}_{}_{}_{}.png'.format(x, y, self.pw, self.patch_size_10X)
        patch = self.oslide.read_region((x, y), 0, (pw_x, pw_y))
        patch = patch.resize((int(self.patch_size_10X * pw_x / self.pw), int(self.patch_size_10X * pw_y / self.pw)),
                             Image.ANTIALIAS)

        return np.array(patch)[:, :, :3], fname

    def has_next(self):
        return self.index < len(self.coors) + len(self.coors_partial) - 1

    def next_patch(self):
        coors = self.coors_partial + self.coors
        patch, fname = self.extract_patch(coors[self.index])
        self.index += 1
        return patch, fname


class data_loader_WSI(Dataset):
    def __init__(self, patch_extraction_instance, isComplete=True):
        self.patch_extraction = patch_extraction_instance        # patch_extraction instance
        if isComplete:
            self.coors = patch_extraction_instance.coors
        else:
            self.coors = patch_extraction_instance.coors_partial

        self.transform = get_data_transforms()['val']

    def __getitem__(self, index):
        img, fname = self.patch_extraction.extract_patch(self.coors[index])

        img = Image.fromarray(img.astype(np.uint8), 'RGB')
        img = self.transform(img)

        return img, fname

    def __len__(self):
        return len(self.coors)




