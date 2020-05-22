import cv2
import numpy as np

def thresholding_pipeline(img_cv):
    # img_inv = cv2.bitwise_not(img_cv[:,:,0])
    thr1 = cv2.adaptiveThreshold(img_cv[:,:,2],255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                                cv2.THRESH_BINARY,151,-51)
    # thr2 = cv2.adaptiveThreshold(img_inv,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #                             cv2.THRESH_BINARY,151,-21)
    img_hls = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HLS)
    thr2 = np.zeros(img_hls.shape[:-1], dtype=np.uint8)
    thr2[:] = 0
    thr2[(img_hls[:,:,2] > 100)] = 255
    thr2[(img_hls[:,:,0] > 28)|(img_hls[:,:,0] <18)]=0
    #thr2=0
    thr = thr1 | thr2
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(thr,kernel,iterations = 3)
    erosion = cv2.erode(dilation,kernel,iterations = 4)
    return erosion