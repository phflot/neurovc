import yolov5_face.detect_face as yf
import torch
import cv2
from neurovc.util import normalize_color
import numpy as np


class LandmarkWrapper:
    def process(self, img):
        return self.get_landmarks(img), None


class TFWLandmarker(LandmarkWrapper):
    def __init__(self, weights_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = yf.load_model(weights_path, self.device)

    def get_landmarks(self, img):
        img = normalize_color(img, color_map=cv2.COLORMAP_BONE)
        results = yf.detect_landmarks(self.model, img, self.device)
        if len(results) == 0:
            return np.full((5, 2), -1)
        lm = results[0]['landmarks']
        lm = np.array(lm).reshape((-1, 2))
        return lm
