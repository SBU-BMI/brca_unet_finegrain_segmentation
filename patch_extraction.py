import numpy as np
import openslide
from PIL import Image
import multiprocessing as mp


def unwrap_self_extract_patch(arg, **kwarg):
    return patch_extraction.extract_patch(*arg, **kwarg)


class patch_extraction:
    def __init__(self, slide_name, patch_size_10X=1000):
        self.slide_name = slide_name
        self.patch_size_10X = patch_size_10X
        self.oslide, self.pw, self.width, self.height = self.get_oslide()
        self.coors = self.get_coors()
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
                    coors.append((x, y, pw_x, pw_y))

        return coors

    def extract_patch(self, corr):
        x, y, pw_x, pw_y = corr

        fname = '{}_{}_{}_{}.png'.format(x, y, self.pw, self.patch_size_10X)
        patch = self.oslide.read_region((x, y), 0, (pw_x, pw_y))
        patch = patch.resize((int(self.patch_size_10X * pw_x / self.pw), int(self.patch_size_10X * pw_y / self.pw)),
                             Image.ANTIALIAS)

        return np.array(patch)[:, :, :3], fname

    def has_next(self):
        return self.index < len(self.coors) - 1

    def next_patch(self):
        patch, fname = self.extract_patch(self.coors[self.index])
        self.index += 1
        return patch, fname

    def next_batch(self, num_patches=16):
        if self.index + num_patches >= len(self.coors):
            num_patches = len(self.coors) - self.index - 1

        coor_list = self.coors[self.index:self.index + num_patches]
        pool = mp.Pool(processes=num_patches)
        results = pool.map(unwrap_self_extract_patch, zip([self]*len(coor_list), coor_list))
        pool.close()

        self.index += num_patches

        return results







