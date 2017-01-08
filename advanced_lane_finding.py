##############################  Import Some packages  ############################
import csv
import sys
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from sklearn import preprocessing
from keras.optimizers import SGD
from keras.optimizers import Adam
import hashlib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
import math
from skimage import color
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import model_from_json
import cv2
from moviepy.editor import VideoFileClip
from IPython.display import HTML
##############################  ##### 
# prepare objects
objpoints = []
imgpoints = []
mtx = []
dist = []


#Prepare object points, like (0,0,0), (1,0,0), (2,0,0) .....,(7,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)                # x, y coordinates

def draw_corners(fname, nx, ny):
	img = cv2.imread(fname)

	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Find the chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
	#print(ret)
	# If found, draw corners
	if ret == True:
	    # Draw and display the corners
	    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
	    plt.imshow(img)
	return ret, corners

# Prepare Objpoints and imagepoints
###  use glob.glob to read more images
with open("./camera_cal.csv") as csv_file:
	fileread = csv.reader(csv_file)
	for row in fileread:	
		nx = int(row[1])
		ny = int(row[2])
		ret, cor = draw_corners(row[0], nx, ny)
		if ret == True:
			objpoints.append(objp)
			imgpoints.append(cor)
		#plt.show()

### Use objpoints and imgpoints to undistort test images images

def  undistort(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
	
	return cv2.undistort(img, mtx, dist, None, mtx)

with open("./camera_cal.csv") as csv_file:
	fileread = csv.reader(csv_file)
	for row in fileread:	
		nx = int(row[1])
		ny = int(row[2])
		img = cv2.imread(row[0])
		dst = undistort(img)
		#plt.imshow(dst)
		#plt.show()
################################## direction Threshold ######################
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients    
    sobel_x_abs = np.absolute(sobel_x)
    sobel_y_abs = np.absolute(sobel_y)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    sobel_x_y = np.arctan2(sobel_y_abs, sobel_x_abs)
    # 5) Create a binary mask where direction thresholds are met
    sobel_sc = sobel_x_y
    sobel_mask= np.zeros_like(sobel_sc)
    sobel_mask[(sobel_sc >= thresh[0]) & (sobel_sc <= thresh[1])] = 1
    binary_output = sobel_mask
    # 6) Return this mask as your binary_output image    
    return binary_output

################################# Gray Image Thresholding ##################
def gray_threshold(img, thresh=(180, 255)):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	binary = np.zeros_like(gray)
	binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1
	return binary

################################# Green Image Thresholding ##################
def g_threshold(img, thresh=(180, 255)):
	G = img[:,:,1]
	binary = np.zeros_like(G)
	G = img[:,:,1]
	binary[(G > thresh[0]) & (G <= thresh[1])] = 1
	return binary

################################# Gray Image Thresholding ##################
def hls_s_threshold(img, thresh=(180, 255)):
	hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	H = hls[:,:,0]
	L = hls[:,:,1]
	S = hls[:,:,2]
	binary = np.zeros_like(S)
	binary[(S > thresh[0]) & (S <= thresh[1])] = 1
	return binary

############################# Mask the image ###############################
def mask_the_image(img):
	left_bottom = [0, img.shape[0]]
	right_bottom = [img.shape[1], img.shape[0]]
	apex = [img.shape[1]/2, img.shape[0]/2]
	### take empty container
	region_select = np.copy(img)
	# Fit lines (y=Ax+B) to identify the  3 sided region of interest
	# np.polyfit() returns the coefficients [A, B] of the fit
	fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
	fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
	fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

	# Find the region inside the lines
	XX, YY = np.meshgrid(np.arange(0, img.shape[1]), np.arange(0, img.shape[0]))
	region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))

	# Color pixels red which are inside the region of interest
	region_select[~region_thresholds] = [255, 0, 0]

	# Display the image
	return region_select

########################## Canny Edge Detection ##################
def canny_detect(img, lw_thr=50, high_thr=150):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	kernel_size = 5
	blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
	edges = cv2.Canny(blur_gray, lw_thr, high_thr)
	# Display the image
	return edges

########################## Magnitude Threshold ####################
def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobel_x_y = np.sqrt((cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)**2) + (cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)**2))
    # 3) Take the absolute value of the derivative or gradient
    sobel_abs = np.absolute(sobel_x_y)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel_sc = np.uint8(255*(sobel_abs/np.max(sobel_abs)))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    sobel_mask= np.zeros_like(sobel_sc)
    # 6) Return this mask as your binary_output image
    sobel_mask[(sobel_sc >= thresh[0]) & (sobel_sc <= thresh[1])] = 1
    binary_output = sobel_mask
    return binary_output

########################## Percpective Transform ###################
def pers_trans(img):
	img_size = (img.shape[1], img.shape[0])
	top_lft_org = [520, 490]
	top_rgt_org = [760, 490]
	bot_lft_org = [229,719]
	bot_rgt_org = [1220,719]
	src = np.float32([top_rgt_org, bot_rgt_org, bot_lft_org, top_lft_org ])
	top_lft_map = [229, 0]
	top_rgt_map = [1182, 0]
	bot_lft_map = [229,719]
	bot_rgt_map = [1220,719]
	dst = np.float32([top_rgt_map, bot_rgt_map, bot_lft_map, top_lft_map ])
	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)
	warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
	return warped, Minv
###################### Hogh transform ######################
def draw_lines(img, lines, color=[255, 0, 0], thickness=1):
    imshape = img.shape
    for line in lines:
        
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, initial_img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros(initial_img.shape, dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def hogh_apply(img, min_line_len=40, max_line_gap=150):
	rho = 2 #distance resolution in pixels of the Hough grid
	theta = np.pi/180 #angular resolution in radians of the Hough grid
	threshold = 20     #minimum number of votes (intersections in Hough grid cell)
	line_image = hough_lines(dir_binary,  dir_binary, rho, theta, threshold, min_line_len, max_line_gap)
	return line_image
#################### Use histogram to find lines ########################################
def non_zero_neigh_l(i, l_pix):
    list = []
    if (i < len(l_pix)/2):
        list = range(i, len(l_pix))
    else:
        list = range(i, 0, -1)
    
    l = []
    a = []
    for l in list:
        if(l_pix[l] !=0):
            a = l_pix[l]
            break;
    return a

def non_zero_neigh_r(i, r_pix, val):
    list = []
    if (i < len(r_pix)/2):
        list = range(i, len(r_pix))
    else:
        list = range(i, 0, -1)
    l = [] 
    a = []
    for l in list:
        if(r_pix[l] != val):
            a = r_pix[l]
            break;
    return a

def add_av(left_pix, right_pix, val):
    for i in range(0, len(left_pix)):
        if(left_pix[i] == 0):
            left_pix[i] = non_zero_neigh_l(i, left_pix)

    for i in range(0, len(right_pix)):
        if(right_pix[i] == val):
            right_pix[i] = non_zero_neigh_r(i, right_pix, val)
            
    return left_pix, right_pix

def find_lanes_hist(dir_binary):
    line_mask = np.zeros_like(dir_binary)
    left_pix = []
    right_pix = []
    for i in range(0, dir_binary.shape[0]-1):
        if (i < (dir_binary.shape[0]/2)):
            histogram = np.sum(dir_binary[i:i+50,:], axis=0)
        else:
            histogram = np.sum(dir_binary[i-50:i,:], axis=0)
        
        A = np.argmax(histogram[1:len(histogram)/2])
        B = np.argmax(histogram[len(histogram)/2:])        
        #print(A,B)
                    
        line_mask[i, A] = 1
        left_pix.append(A)            
        line_mask[i, B+len(histogram)/2] = 1
        right_pix.append(B+len(histogram)/2)
        
    left_pix, right_pix = add_av(np.array(left_pix), np.array(right_pix), len(histogram)/2)        
    return np.array(left_pix), np.array(right_pix), line_mask

################## Fit polynominal ###################
def fit_pol(leftx, rightx, yvals):
    left_fit = np.polyfit(yvals, leftx, 2)
    left_fitx = left_fit[0]*yvals**2 + left_fit[1]*yvals + left_fit[2]
    right_fit = np.polyfit(yvals, rightx, 2)
    right_fitx = right_fit[0]*yvals**2 + right_fit[1]*yvals + right_fit[2]
    # Plot up the lane data
    plt.plot(leftx, yvals, 'o', color='red')
    plt.plot(rightx, yvals, 'o', color='blue')
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.plot(left_fitx, yvals, color='green', linewidth=3)
    plt.plot(right_fitx, yvals, color='green', linewidth=3)
    plt.gca().invert_yaxis() # to visualize as we do the images
    y_eval = np.max(yvals)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) \
                             /np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) \
                                /np.absolute(2*right_fit[0])
    plt.text(0,50, left_curverad, bbox=dict(facecolor='red', alpha=0.5))
    plt.text(600,50,  right_curverad, bbox=dict(facecolor='red', alpha=0.5))
    ### Map to the real world problem
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension

    left_fit_cr = np.polyfit(yvals*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(yvals*ym_per_pix, rightx*xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) \
                             /np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) \
                                /np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 3380.7 m    3189.3 m
    plt.text(0,50, left_curverad, bbox=dict(facecolor='red', alpha=0.5))
    plt.text(600,50,  right_curverad, bbox=dict(facecolor='red', alpha=0.5))
    return left_fitx, right_fitx, yvals, left_curverad, right_curverad
############## Redraw on the original image #################
def redraw(warped, undist, Minv, left_fitx, right_fitx, yvals):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

###############
def find_ln_overlay(dir_binary, undst, Minv):
    left_pix, right_pix, lane_detect = find_lanes_hist(dir_binary)
    yval = np.array(range(0,dir_binary.shape[0]-1))
    left_fitx, right_fitx, yvals, left_curverad, right_curverad = fit_pol(left_pix, right_pix, yval)
    #plt.show()
    lane_detect = redraw(dir_binary , undst, Minv, left_fitx, right_fitx, yvals)
    return lane_detect

def pre_proc(img):
    undst = undistort(img)
    dst = mask_the_image(undst)
    edges = canny_detect(dst, lw_thr=20, high_thr=100)
    dir_binary, Minv = pers_trans(edges)
    dir_binary = hogh_apply(dir_binary)
    return find_ln_overlay(dir_binary, undst, Minv)

################################## Take a test image and undistort/test it ##############
img = cv2.imread("./test_images/test1.jpg")
undst = undistort(img)
plt.imshow(cv2.cvtColor(undst, cv2.COLOR_BGR2RGB))
plt.title('Undistortion')
plt.show()
dst = mask_the_image(undst)

dir_binary = dir_threshold(dst, sobel_kernel=15, thresh=(0.7, 1.2))
dir_binary, Minv = pers_trans(dir_binary)
lane_detect = find_ln_overlay(dir_binary, undst, Minv)
plt.imshow(lane_detect, cmap='gray')
plt.title('Direction Threshold')
plt.show()

dir_binary = hls_s_threshold(dst, thresh=(180, 255))
dir_binary, Minv = pers_trans(dir_binary)
lane_detect = find_ln_overlay(dir_binary, undst, Minv)
plt.imshow(lane_detect, cmap='gray')
plt.title('HLS Saturation Threshold')
plt.show()

edges = canny_detect(dst, lw_thr=20, high_thr=100)
dir_binary, Minv = pers_trans(edges)
plt.imshow(dir_binary, cmap='gray')
plt.title('Canny Result')
plt.show()
dir_binary = hogh_apply(dir_binary)
lane_detect = find_ln_overlay(dir_binary, undst, Minv)
plt.imshow(lane_detect, cmap='gray')
plt.title('After hogh Transform')
plt.show()

dir_binary = g_threshold(dst, thresh=(180, 255))
dir_binary, Minv = pers_trans(dir_binary)
plt.imshow(dir_binary, cmap='gray')
plt.title('Green Threshold')
plt.show()

plt.figure(1)
dir_binary = mag_thresh(dst, sobel_kernel=15, thresh=(50, 150))
dir_binary, Minv = pers_trans(dir_binary)
plt.imshow(dir_binary, cmap='gray')
plt.title('Magnitude Threshold')

lane_detect = find_ln_overlay(dir_binary, undst, Minv)
plt.show()
plt.imshow(lane_detect, cmap='gray')
plt.title('Lane Lines detected with Magnitude Threshold')
plt.show()

dir_binary = gray_threshold(dst, thresh=(180, 255))
dir_binary, Minv = pers_trans(dir_binary)
plt.imshow(dir_binary, cmap='gray')
plt.title('Gray Threshold')
plt.show()
plt.figure(1)
dir_binary = hogh_apply(dir_binary, 80, 300)
plt.imshow(dir_binary, cmap='gray')
plt.title('hogh Transform after gray')

plt.figure(2)

lane_detect = find_ln_overlay(dir_binary, undst, Minv)
plt.imshow(lane_detect, cmap='gray')
plt.title('Lane Lines detected')
plt.show()


white_output = 'overlayed.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(pre_proc)
white_clip.write_videofile(white_output, audio=False)