import numpy as np
import cv2


ym_per_pix = 30/720
xm_per_pix = 3.7/700


def get_poly_curvature(A, B, C):
    def get_curvature(y):
        dxdy = 2*A*y + B
        dxdy2 = 2*A
        R = ((1 + dxdy ** 2.0) ** 1.5) / abs(dxdy2)
        return R
    return get_curvature

def find_lane_pixels(binary_warped, nwindows=9, margin=100, minpix=50):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = np.int(binary_warped.shape[0]//nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # cv2.rectangle(
        #     out_img,
        #     (win_xleft_low, win_y_low),
        #     (win_xleft_high, win_y_high),
        #     (0,255,0),
        #     2
        # ) 
        # cv2.rectangle(
        #     out_img,
        #     (win_xright_low, win_y_low),
        #     (win_xright_high, win_y_high),
        #     (0,255,0),
        #     2
        # )

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

left_fits = []
right_fits = []

def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700

    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    except:
        return out_img

    left_fits.append(left_fit)
    right_fits.append(right_fit)

    cur_left_fits = np.array(left_fits[-5:])
    cur_right_fits = np.array(right_fits[-5:])
    #print(cur_right_fits)
    left_fit = np.median(cur_left_fits, axis=0)
    right_fit = np.median(cur_right_fits, axis=0)
    #print(right_fit)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    
    # print(left_fit)
    # print(right_fit)

    h, w = binary_warped.shape
    lane_color = [0, 255, 0]
    for y in range(h):
        leftx = int(left_fit[0]*y**2 + left_fit[1]*y + left_fit[2])
        rightx = int(right_fit[0]*y**2 + right_fit[1]*y + right_fit[2])
        out_img[y, leftx:rightx] = lane_color
        # for i in range(leftx, rightx):
        #     if i >= 0 and i < w:
        #         out_img[y, i] = lane_color
        # for i in range(5):
        #     if leftx+i >= 0 and leftx+i < w:
        #         out_img[y, leftx+i] = lane_color
        #     if rightx+i >= 0 and rightx+i < w:
        #         out_img[y, rightx+i] = lane_color
        #     if leftx-i >= 0 and leftx-i < w:
        #         out_img[y, leftx-i] = lane_color
        #     if rightx-i >= 0 and rightx-i < w:
        #         out_img[y, rightx-i] = lane_color

    cary = h
    leftlanex = (left_fit[0]*cary**2 + left_fit[1]*cary + left_fit[2])
    rightlanex = (right_fit[0]*cary**2 + right_fit[1]*cary + right_fit[2])
    carx = (leftlanex+rightlanex) / 2
    center_offset = (carx - (w/2)) * xm_per_pix

    return out_img, measure_curvature_pixels(cary, left_fit_cr, right_fit_cr), center_offset

def measure_curvature_pixels(y_eval, left_fit, right_fit):
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    return left_curverad, right_curverad