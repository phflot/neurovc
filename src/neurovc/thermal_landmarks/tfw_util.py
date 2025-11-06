import os
from pathlib import Path

import cv2
import numpy as np

from neurovc.util import normalize_color

try:
    import torch

    HAS_TORCH = True
except ModuleNotFoundError:
    torch = None
    HAS_TORCH = False

try:
    import yolov5_face.detect_face as yf

    HAS_YOLOV5_FACE = True
except ModuleNotFoundError:
    yf = None
    HAS_YOLOV5_FACE = False

try:
    import requests

    HAS_REQUESTS = True
except ModuleNotFoundError:
    requests = None
    HAS_REQUESTS = False

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ModuleNotFoundError:
    tqdm = None
    HAS_TQDM = False


class _ModelDownloader:
    def __init__(self, model_name, save_dir="~/.neurovc/models"):
        if not HAS_REQUESTS:
            raise ImportError(
                "requests is required to download models. Install the landmark extra: 'pip install neurovc[landmark]'"
            )
        self.model_name = model_name
        self.save_dir = Path(os.path.expanduser(save_dir))
        self.file_id = _file_id_map.get(model_name)
        if not self.file_id:
            raise ValueError(
                f"Model name '{model_name}' is not valid. Check the file_id_map."
            )
        self.model_url = (
            f"https://drive.google.com/uc?export=download&id={self.file_id}"
        )
        self.model_path = self.save_dir / f"{model_name}.pt"

    def download_model(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.model_path.exists():
            return self.model_path

        print(f"Downloading {self.model_name}...")
        response = requests.get(self.model_url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get("content-length", 0))
            progress_iter = None
            if HAS_TQDM and total_size:
                progress_iter = tqdm(
                    desc=f"Downloading {self.model_name}",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                )
            with open(self.model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    if progress_iter is not None:
                        progress_iter.update(len(chunk))
            if progress_iter is not None:
                progress_iter.close()
            print(f"Model downloaded to {self.model_path}")
        else:
            raise Exception(f"Failed to download model: {response.status_code}")
        return self.model_path


def _prepare_model(model_name="YOLOv5n-Face"):
    downloader = _ModelDownloader(model_name)
    model_path = downloader.download_model()
    return model_path


class LandmarkWrapper:
    def process(self, img):
        return self.get_landmarks(img), None


class TFWLandmarker(LandmarkWrapper):
    def __init__(self, model_name="YOLOv5n-Face"):
        if not HAS_TORCH:
            raise ImportError(
                "torch is required for TFWLandmarker. Install the torch extra: 'pip install neurovc[torch]'"
            )
        if not HAS_YOLOV5_FACE:
            raise ImportError(
                "yolov5-face is required for TFWLandmarker. Install the landmark extra: 'pip install neurovc[landmark]'"
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = _prepare_model(model_name)
        self.model = yf.load_model(model_path, self.device)

    def detect(self, img):
        img = normalize_color(img, color_map=cv2.COLORMAP_BONE)
        results = yf.detect_landmarks(self.model, img, self.device)
        return results

    def get_landmarks(self, img):
        results = self.detect(img)
        if len(results) == 0:
            return np.full((5, 2), -1)
        lm = results[0]["landmarks"]
        lm = np.array(lm).reshape((-1, 2))
        return lm


_file_id_map = {
    "YOLOv5n": "1PLUq7WbOWS7Ve2VKW7_WBkC3Uksje8Fx",
    "YOLOv5n6": "1wV9t5uH_eiy7WaHdQdWnbeEIijuDAdKI",
    "YOLOv5s": "1IdsdR1-qUeRo5EKQJzGQmRDi2SrMXJG5",
    "YOLOv5s6": "1YZX3t7cSPnWWoic7oJo86ljBQgE5PPb2",
    "YOLOv5n-Face": "1vXk9P3CfhUtRBGI44SqWbuiTJ7rAI4hP",
}


__all__ = [
    "HAS_TORCH",
    "HAS_YOLOV5_FACE",
    "HAS_REQUESTS",
    "HAS_TQDM",
    "TFWLandmarker",
    "LandmarkWrapper",
]
