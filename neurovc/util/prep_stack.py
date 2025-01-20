from neurovc.util.IO_util import *


def __preproc1(frame_cel):
    frame_cel[frame_cel < 20] = 20
    frame_cel[frame_cel > 45] = 45
    frame_cel = frame_cel + 2 * (frame_cel - imgaussfilt(frame_cel, 5))
    return cv2.normalize(frame_cel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def __preproc2(frame_cel):
    frame_cel[frame_cel < 20] = 20
    frame_cel[frame_cel > 45] = 45
    frame_cel = frame_cel + 4 * (frame_cel - cv2.GaussianBlur(frame_cel, (15, 15), 5))
    return cv2.normalize(frame_cel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def __preproc3(frame_cel):
    return 1 - __preproc1(frame_cel)


def __preproc4(frame_cel):
    return 1 - __preproc2(frame_cel)


def get_preprocessing_stack():
    return [__preproc1, __preproc2, __preproc3, __preproc4]