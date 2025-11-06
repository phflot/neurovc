from threading import Thread
from collections import deque

try:
    from ximea.xiapi import Camera, Image, Xi_error

    HAS_XIMEA = True
except ModuleNotFoundError:
    Camera = Image = Xi_error = None  # type: ignore[assignment]
    HAS_XIMEA = False


def init_cam(cam):
    # from ximea.xiapi import Camera, Image, Xi_error
    # cam = Camera()

    cam.set_exposure(10000)
    # cam.set_imgdataformat('XI_RGB24')
    # cam.set_limit_bandwidth(cam.get_limit_bandwidth())
    cam.set_imgdataformat("XI_RAW8")
    # cam.set_trigger_source('XI_TRG_SOFTWARE')
    # cam.set_trigger_source('XI_TRG_SOFTWARE')
    # cam.set_acq_timing_mode(1)


class StereoConfig:
    def __init__(self, left_id="18950051", right_id="06954151"):
        self.left_id = left_id
        self.right_id = right_id

    # todo: replace with the serial number of the left camera in the stereo setup
    def get_left_id(self):
        return self.left_id

    # todo: replace with the serial number of the right camera in the stereo setup
    def get_right_id(self):
        return self.right_id


class GenericCamera:
    def __init__(self):
        pass

    def get_image(self):
        pass


class XimeaCamera(GenericCamera):
    def __init__(self, sn):
        if not HAS_XIMEA:
            raise ImportError(
                "ximea.xiapi is required for XimeaCamera. Install the proprietary SDK to enable this feature."
            )
        self.cam = Camera()
        self.sn = sn
        self.cam.open_device_by_SN(sn)
        # self.cam.open_device_by("XI_OPEN_BY_USER_ID", sn)
        init_cam(self.cam)
        self.cam.start_acquisition()
        self.__image = Image()

    def get_image(self):
        try:
            self.cam.get_image(self.__image)
            image = self.__image.get_image_data_numpy()
            ts = 1000000 * self.__image.tsSec + self.__image.tsUSec
            n_frame = self.__image.acq_nframe
        except Exception:
            return (None, None, None)

        return n_frame, ts, image


class FrameGrabber(Thread):
    def __init__(self, cam):
        Thread.__init__(self)
        self.cam = cam
        self.__callbacks = []
        self.frame_deque = deque()

    def run(self):
        while True:
            n_frame, ts, frame = self.cam.get_image()
            if frame is None:
                continue
            self.__callback(n_frame, ts, frame)
            result_tuple = (n_frame, ts, frame)
            self.frame_deque.append(result_tuple)

    def add_callback(self, callback):
        self.__callbacks.append(callback)

    def remove_callback(self, callback):
        self.__callbacks.remove(callback)

    def __callback(self, n_frame, ts, frame):
        for f in self.__callbacks:
            f(n_frame, ts, frame)


class MultimodalWorker(Thread):
    def __init__(self):
        Thread.__init__(self)
        self._cams = []

    def add_cam(self, cam):
        self._cams.append(cam)


__all__ = [
    "HAS_XIMEA",
    "init_cam",
    "StereoConfig",
    "GenericCamera",
    "XimeaCamera",
    "FrameGrabber",
    "MultimodalWorker",
]
