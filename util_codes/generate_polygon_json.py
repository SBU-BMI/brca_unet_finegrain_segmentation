import json
import numpy as np
from pycocotools import mask
from skimage import measure
from util_codes import patch_extraction
import glob, os, cv2
import time


def get_json_template():
    json = {
        "polygon": [],
        "area": 0
    }
    return json


class generate_polygon_json:
    def __init__(self, wsi_path, wsi_out):
        self.wsi_path = wsi_path
        self.wsi_out = wsi_out
        self.patch_extraction = patch_extraction(wsi_path)
        self.oslide, self.width, self.height = self.get_patch_extraction_info()

    def get_patch_extraction_info(self):
        return self.patch_extraction.oslide, self.patch_extraction.width, self.patch_extraction.height

    def get_pngs(self):
        png_fns = glob.glob(os.path.join(self.wsi_out, '*.png'))
        return png_fns

    def convert_polygon(self, p, top_left=(0, 0), ratio=1):
        x0, y0 = top_left
        w, h = self.width, self.height
        out = []
        for i in range(0, len(p), 2):
            x, y = p[i] * ratio, p[i + 1] * ratio
            out.append([(x + x0) / w, (y + y0) / h])
        return out

    def generate_json_one_patch(self, fn):
        # fn: svs_tcga_seer_brca/TCGA-A8-A06N-01Z-00-DX1.svs/8603_51613_4301_1000.png
        patch_name = fn.rstrip('/').split('/')[-1]  # 8603_51613_4301_1000.png
        patch_info = patch_name[:-4].split('_')     # ['8603', '51613', '4301', '1000']
        top_left = (int(patch_info[0]), int(patch_info[1]))
        ratio = float(patch_info[2])/float(patch_info[3])

        binary_mask = cv2.imread(fn, 0)
        fortran_binary_mask = np.asfortranarray(binary_mask)
        encoded_mask = mask.encode(fortran_binary_mask)
        area = mask.area(encoded_mask)
        contours = measure.find_contours(binary_mask, 0.5)

        json = get_json_template()
        json["area"] = area.tolist()

        for contour in contours:
            contour = np.flip(contour, axis=1)
            polygon = contour.ravel().tolist()
            if len(polygon) >= 8:   # require at least 4 points to form a polygon
                polygon = self.convert_polygon(polygon, top_left, ratio)    # convert from 1 dim to (x, y) pair
                polygon.append(polygon[0])      # to close the loop of the polygon
                json["polygon"].append(polygon)

        return json

    def main(self):
        png_fns = self.get_pngs()
        json_wsi = get_json_template()
        print('Generating json file for: {}, {} patches'.format(self.wsi_out, len(png_fns)))
        start = time.time()

        for fn in png_fns:
            json_patch = self.generate_json_one_patch(fn)
            json_wsi["polygon"].extend(json_patch["polygon"])
            json_wsi["area"] += json_patch["area"]

        wsi_name = self.wsi_out.rstrip('/').split('/')[-1]
        json_fn = os.path.join(self.wsi_out, wsi_name + '.json')
        with open(json_fn, 'w') as fp:
            json.dump(json_wsi, fp)

        print('Done generating json file: {}, {} patches. Time: {:.2f}mins'.format(self.wsi_out, len(png_fns), (time.time() - start)/60))
        return json_wsi






