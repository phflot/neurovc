# from here: https://github.com/apple2373/mediapipe-facemesh
# and here: https://github.com/google/mediapipe/issues/1698
import json
import numpy as np
from scipy.interpolate import griddata, LinearNDInterpolator, NearestNDInterpolator
from neurovc.momag import warp_image_backwards


class UVMapperScipy:
    def __init__(self, image_size=(256, 256)):
        uv_map_dict = json.load(open("uv_map.json"))
        self.uv_map = np.array([(uv_map_dict["u"][str(i)], uv_map_dict["v"][str(i)]) for i in range(468)])
        self.image_size = image_size
        self.ref = self.uv_map * np.array(image_size).reshape((1, 2))
        self.coords = np.meshgrid(np.arange(image_size[1], dtype=float), np.arange(image_size[0], dtype=float))

    def __call__(self, img, points):
        n, m = self.image_size
        ref = self.ref
        x, y = self.coords

        flow_init = points - ref

        w = LinearNDInterpolator((ref[:, 0], ref[:, 1]), flow_init, fill_value=np.float32('inf'))
        w = w(x.flatten(), y.flatten()).reshape((n, m, 2)).astype(np.float32)

        return warp_image_backwards(img, w)
