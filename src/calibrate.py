from glob import glob
from os import path

import numpy as np
import cv2


def calibrate_camera(calib_imgs_dir, nx, ny):
    objp = np.zeros((nx*ny, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    calib_imgs = glob(path.join(calib_imgs_dir, 'calibration*.jpg'))
    h, w, _ = cv2.imread(calib_imgs[0]).shape
    for img_path in calib_imgs:
        img_cv = cv2.imread(img_path)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        ret, corner_coords = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corner_coords)

    _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
    return mtx, dist

def undistort(img_cv, K, D):
    return cv2.undistort(img_cv, K, D, None, K)


if __name__ == '__main__':
    K, D = calibrate_camera('camera_cal', nx=9, ny=6)
    for img_path in glob(path.join('camera_cal', 'calibration*.jpg')):
        img_cv = cv2.imread(img_path)
        undist_cv = undistort(img_cv, K, D)
        cv2.imshow('Normal', img_cv)
        cv2.imshow('Undistorted', undist_cv)
        cv2.waitKey(0)