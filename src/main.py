import sys
import io

import cv2
import numpy as np
import matplotlib.pyplot as plt

import transform
import threshold
import histogram
import line_poly_finder
import calibrate

RESIZE = (640//2, 360//2)


def get_img_from_fig(fig, dpi=80):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


if __name__ == '__main__':
    vid_path = sys.argv[1]
    cap = cv2.VideoCapture(vid_path)
    idx = 0

    K, D = calibrate.calibrate_camera('camera_cal', nx=9, ny=6)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # if not ((idx > 990) and (idx < 1055)):
        #     idx+=1
        #     continue

        frame = calibrate.undistort(frame, K, D)
        resized_frame = cv2.resize(frame, RESIZE)

        # Display the resulting frame
        # cv2.imshow('frame', resized_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        road_img = transform.transform_road_img(frame)
        cv2.imwrite('prob/{}.jpg'.format(idx), road_img)
        resized_road_img = cv2.resize(road_img, RESIZE)

        # cv2.imshow('road_img', resized_road_img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        thr = threshold.thresholding_pipeline(road_img)
        resized_thr = cv2.resize(thr, RESIZE)
        resized_thr = cv2.cvtColor(resized_thr, cv2.COLOR_GRAY2BGR)
        # cv2.imshow('thr', resized_thr)

        big_img = np.zeros((4*RESIZE[1], RESIZE[0], 3), dtype=np.uint8)
        big_img[0:RESIZE[1],0:RESIZE[0]] = resized_frame
        big_img[RESIZE[1]:2*RESIZE[1],0:RESIZE[0]] = resized_road_img
        big_img[2*RESIZE[1]:3*RESIZE[1],0:RESIZE[0]] = resized_thr

        # hist = histogram.hist(thr)
        # fig = plt.figure(figsize=(16, 9))
        # plt.plot(hist)
        # fig.canvas.draw()
        # plot_img_np = get_img_from_fig(fig)
        # plot_img_np_resized = cv2.resize(plot_img_np, RESIZE)
        # big_img[3*RESIZE[1]:4*RESIZE[1],0:RESIZE[0]] = plot_img_np_resized
        # plt.close(fig)

        lanes, (left_curverad, right_curverad), offset = line_poly_finder.fit_polynomial(thr)
        big_img[3*RESIZE[1]:4*RESIZE[1],0:RESIZE[0]] = cv2.resize(lanes, RESIZE)

        M_inv = transform.get_perspective_transform(lanes, inverse=True)
        lanes_warped_inv = cv2.warpPerspective(
            lanes, M_inv, (lanes.shape[1], lanes.shape[0]), flags=cv2.INTER_LINEAR)

        cv2.addWeighted(lanes_warped_inv, 0.5, frame, 1 - 0.5, 0, frame)
        cv2.putText(
            frame,
            'Curvature: {}m'.format((left_curverad+right_curverad)/2), 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1,
            (255,255,255),
            2
        )
        cv2.putText(
            frame,
            'Center offset: {}m'.format(offset), 
            (10, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1,
            (255,255,255),
            2
        )

        cv2.imwrite('processed/{:05d}.jpg'.format(idx), frame)
        idx+=1
        if idx % 100 == 0:
            print(idx)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
