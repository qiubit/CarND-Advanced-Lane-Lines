import cv2
import numpy as np


# DEFAULT_ROI = [
#     # bottom left/right
#     (200, 720-50),
#     (1280-170, 720-50),
#     # top left/right
#     (640-60, 360+90),
#     (640+100, 360+90),
# ]

DEFAULT_ROI = [
    # bottom left/right
    (255, 690),
    (1050, 690),
    # top left/right
    (590, 455),
    (695, 455),
]


def get_perspective_transform(img_cv, roi=DEFAULT_ROI, offset=300, inverse=False):
    h, w, _ = img_cv.shape
    src = np.array(roi, dtype=np.float32)
    dst = np.array([
        (offset, h),
        (w-offset, h),
        (offset, 0),
        (w-offset, 0)
    ], dtype=np.float32)
    print('src')
    print(src)
    print('dst')
    print(dst)
    if inverse:
        M = cv2.getPerspectiveTransform(dst, src)
    else:
        M = cv2.getPerspectiveTransform(src, dst)
    return M

def transform_road_img(img_cv):
    h, w, _ = img_cv.shape
    M = get_perspective_transform(img_cv)
    return cv2.warpPerspective(img_cv, M, (w, h), flags=cv2.INTER_LINEAR)


if __name__ == '__main__':
    from glob import glob
    from os import path

    test_images = glob(path.join('test_images', '*.jpg'))
    h, w, _ = cv2.imread(test_images[0]).shape
    big_img = np.zeros((h, w//2, 3), dtype=np.uint8)
    for idx, img_path in enumerate(test_images):
        col = idx // 4
        row = idx % 4

        img_cv = cv2.imread(img_path)
        img_cv = transform_road_img(img_cv)
        h, w, _ = img_cv.shape
        cv2.imwrite('warped/{}.jpg'.format(idx), img_cv)
        h = h // 4
        w = w // 4
        big_img[row*h:(row+1)*h,col*w:(col+1)*w] = cv2.resize(img_cv, (w, h))
    cv2.imshow('frame', big_img)
    cv2.waitKey(0)
    cv2.imwrite('output_images/big_warped_img.jpg', big_img)


    for img_path in test_images:
        img_cv = cv2.imread(img_path)
        for coord in DEFAULT_ROI:
            cv2.circle(img_cv, coord, 3, (0, 0, 255), -1)
        cv2.imshow('Frame', img_cv)
        cv2.waitKey(0)
        warped = transform_road_img(img_cv)
        cv2.imshow('Frame', warped)
        cv2.waitKey(0)