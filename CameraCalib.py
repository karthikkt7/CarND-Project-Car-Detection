import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

###Functions for transformation###
def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def bgr2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def bgr2hls(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)	

def bgr_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def cameraCalib(objpoints, imgpoints, shape):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape,None,None)
    return mtx, dist, rvecs, tvecs

def UndistImg(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)
	
#Used from ./examples/example.ipynb
def FindChessCorners(Imgs):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    imgs = []

    # Step through the list and search for chessboard corners
    for img in Imgs:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            #cv2 --> BGR format
            img = bgr2rgb(cv2.drawChessboardCorners(img, (9,6), corners, True))
            imgs.append(img)
            
    return objpoints, imgpoints, imgs

def toParam(left_coeff, right_coeff, w=1280*3.7/700, y_m=720*3/50):
    # computing the curvature radii
    left_curverad_pm = ((1 + (2*left_coeff[0]*y_m + left_coeff[1])**2)**1.5) / (2*left_coeff[0])
    right_curverad_pm = ((1 + (2*right_coeff[0]*y_m + right_coeff[1])**2)**1.5) / (2*right_coeff[0])
    # computing the car's position
    pos_left = left_coeff[0]*y_m**2 + left_coeff[1]*y_m + left_coeff[2]
    pos_right = right_coeff[0]*y_m**2 + right_coeff[1]*y_m + right_coeff[2]
    pos_mid = 0.5 * (pos_right + pos_left)
    # difference to center of the image
    pos_car = w/2 - pos_mid
    # lane distance
    dist_lanes = pos_right - pos_left
    # angles
    alpha_left  = np.arcsin(2*left_coeff[0]*y_m + left_coeff[1])
    alpha_right = np.arcsin(2*right_coeff[0]*y_m + right_coeff[1])
    return pos_car, dist_lanes, alpha_left, alpha_right, left_curverad_pm, right_curverad_pm


def toCoeff(pos_car, dist_lanes, alpha_left, alpha_right, left_curverad_pm, right_curverad_pm, w=1280*3.7/700, y_m=720*3/50):
    sal=np.sin(alpha_left)
    sar=np.sin(alpha_right)
    A_left  = (1+sal**2)**1.5/(2*left_curverad_pm)
    A_right = (1+sar**2)**1.5/(2*right_curverad_pm)
    B_left  = sal - 2*A_left*y_m
    B_right = sar - 2*A_right*y_m
    C_left  = 0.5*(w - dist_lanes) - pos_car - A_left *y_m**2 - B_left*y_m
    C_right = 0.5*(w + dist_lanes) - pos_car - A_right*y_m**2 - B_right*y_m
    left_coeff  = np.array([A_left,  B_left,  C_left])
    right_coeff = np.array([A_right, B_right, C_right])
    return left_coeff, right_coeff

