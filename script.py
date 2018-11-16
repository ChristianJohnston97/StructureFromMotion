#####################################################################

# Structure from motion: Computer Vision Assignment L4

# Author : Christian Johnston, christian.johnston@durham.ac.uk

#####################################################################
# cv 3.4.2
import cv2
import os
import numpy as np
import random
import csv
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#####################################################################

img1 = "./images/OH-sfm-test-sequence-DURHAM-2015-03-27-calibrated/2015-03-27_10-47-08_Seq2/monocular_left_calibrated/Images_cam0_7319.png"
img2 = "./images/OH-sfm-test-sequence-DURHAM-2015-03-27-calibrated/2015-03-27_10-47-08_Seq2/monocular_left_calibrated/Images_cam0_7320.png" 

####################################################################

def siftFeatureDetection(img):
    sift = cv2.xfeatures2d.SIFT_create()
    keyPoints, descriptor = sift.detectAndCompute(img, None)
    return keyPoints, descriptor

####################################################################

def featureMatcher(img1, img2, keypoints1,keypoints2, descriptor1, descriptor2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)

    # need both these list
    good = []
    good2 = []

    # for fundamental matrix
    pts1 = []
    pts2 = []

    # ratio test
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good2.append([m])
            good.append(m)

            # for fundamental matrix
            pts2.append(keypoints2[m.trainIdx].pt)
            pts1.append(keypoints1[m.queryIdx].pt)

    # This is to plot the feature matches
    # matchedFeaturesImage = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, good2, None, flags = 2)
    return good, pts1, pts2

####################################################################
# pts1 and pts2 from ratio test as above
def getFundamentalMatrix(pts1, pts2):        
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.1, 0.99)
    return F, mask

####################################################################
# Eessential matrix = K^ . F . K
def getEssentialMatrix(F):
    # K is the intrinsic camera paramaters
    k = np.array([[1.9304844e+03, 0, 6.14632329e+02], [0, 1.18642523e+03, 4.33180837e+02], [0, 0, 1]])
    
    # F is the fundamental matrix (T does the transpose)
    E = k.T.dot(F).dot(k)
    return E


####################################################################
def getRotationAndTranslation(E, pts_l_norm, pts_r_norm):
    # R = rotation matrix, T = translation matrix
    points, R, t, mask = cv2.recoverPose(E, pts_l_norm, pts_r_norm)
    return R, t

####################################################################

def find_camera_matrices_rt(E, Fmask, pts1, pts2):
    # decompose essential matrix into R, t (See Hartley and Zisserman 9.13)

    U, S, Vt = np.linalg.svd(E)
    W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)

    K = np.array([[1.9304844e+03, 0, 6.14632329e+02], [0, 1.18642523e+03, 4.33180837e+02], [0, 0, 1]])
    K_inv = np.linalg.inv(K)

    # iterate over all point correspondences used in the estimation of the fundamental matrix]
    first_inliers = []
    second_inliers = []
    for i in range(len(Fmask)):
        if Fmask[i]:
            # normalize and homogenize the image coordinates
            first_inliers.append(K_inv.dot([pts1[i][0], pts1[i][1], 1.0]))
            second_inliers.append(K_inv.dot([pts2[i][0],pts2[i][1], 1.0]))

    # Determine the correct choice of second camera matrix
    # First choice: R = U * Wt * Vt, T = +u_3

    R = U.dot(W).dot(Vt)
    T = U[:, 2]

    if not in_front_of_both_cameras(first_inliers, second_inliers, R, T):
        # Second choice: R = U * W * Vt, T = -u_3
        R = U.dot(W).dot(Vt)
        T = - U[:, 2]
            
    if not in_front_of_both_cameras(first_inliers, second_inliers, R, T):
        # Third choice: R = U * Wt * Vt, T = u_3
        R = U.dot(W.T).dot(Vt)
        T = U[:, 2]
            
    if not in_front_of_both_cameras(first_inliers, second_inliers, R, T):
        # Fourth choice: R = U * Wt * Vt, T = -u_3
        R = U.dot(W.T).dot(Vt)
        T = - U[:, 2]
            
    match_inliers1 = first_inliers
    match_inliers2 = second_inliers
    return R, T, match_inliers1, match_inliers2

####################################################################
def in_front_of_both_cameras(first_points, second_points, rot, trans):
        rot_inv = rot
        for first, second in zip(first_points, second_points):
            first_z = np.dot(rot[0, :] - second[0]*rot[2, :], trans) / np.dot(rot[0, :] - second[0]*rot[2, :], second)
            first_3d_point = np.array([first[0] * first_z, second[0] * first_z, first_z])
            second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T, trans)
            if first_3d_point[2] < 0 or second_3d_point[2] < 0:
                return False
            return True

####################################################################
def triangluation(match_inliers1, match_inliers2, R, T):
    # first camera is no translation or rotation
    Rt1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    # Second camera consists of R, T
    Rt2 = np.hstack((R, T.reshape(3, 1)))

    first_inliers = np.array(match_inliers1).reshape(-1,3)[:, :2]
    second_inliers = np.array(match_inliers2).reshape(-1,3)[:, :2]

    # This returns the trianglulated real-world points using 4D homogeneous coordinates 
    pts4D = cv2.triangulatePoints(Rt1, Rt2, first_inliers.T, second_inliers.T).T

    pts3d = pts4D[:, :3]/np.repeat(pts4D[:,3], 3).reshape(-1,3)
    return pts3d

####################################################################
def plot3dPointCloud(pts3D):
    Ys = pts3D[:,0]
    Zs = pts3D[:,1]
    Xs = pts3D[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(Xs, Ys, Zs, c='r', marker='o')
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_zlabel('X')
    plt.title('3D point cloud: Use pan axes button below to inspect')
    plt.show()
    return

####################################################################
# Rectify the images
# Taken from Toby Breckon GitHub: python-examples-cv/stereo_sgbm.py 
def rectifyImages(img1, img2, R, T):

    # d and K found from files on DUO
    K = np.array([[1.9304844e+03, 0, 6.14632329e+02], [0, 1.18642523e+03, 4.33180837e+02], [0, 0, 1]])
    d = np.array([-4.8441890104482732e-001, 3.1770182182461387e-001, 4.8167296939537890e-003, 5.9334794668205733e-004,
-1.4902486951308128e-001]).reshape(1,5)
    
    RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(K, d, K, d, img1.shape[:2], R, T, alpha=1)
    #RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(K, d, K, d, img1.shape[:2], R, T, alpha=-1)
    # Toby sets alpha to -1, book sets to 1

    mapL1, mapL2 = cv2.initUndistortRectifyMap(K, d, RL, K, img1.shape[:2], cv2.CV_32F);
    mapR1, mapR2 = cv2.initUndistortRectifyMap(K, d, RR, K, img2.shape[:2], cv2.CV_32F);

    img_rect_1 = cv2.remap(img1, mapL1, mapL2, cv2.INTER_LINEAR);
    img_rect_2 = cv2.remap(img2, mapR1, mapR2, cv2.INTER_LINEAR);
    return img_rect_1, img_rect_2, Q

####################################################################
# Check rectification
def checkRectify(img_rect_1, img_rect_2):
    total_size = (max(img_rect_1.shape[0],img_rect_2.shape[0]), img_rect_1.shape[1] + img_rect_2.shape[1], 3)
    img = np.zeros(total_size, dtype=np.uint8)
    img[:img_rect_1.shape[0], :img_rect_1.shape[1]] = img_rect_1
    img[:img_rect_2.shape[0], img_rect_1.shape[1]:] = img_rect_2
    for i in range(20, img.shape[0], 25):
        cv2.line(img, (0, i), (img.shape[1], i), (255, 0, 0))
        cv2.imshow('imgRectified', img)
    return

####################################################################
    
def computeStereoDepth(img_rect1, img_rect2):
    max_disparity = 128
    stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);
    disparity = stereoProcessor.compute(img_rect1,img_rect2)
    cv2.filterSpeckles(disparity, 0, 40, max_disparity)
    _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO)
    disparity_scaled = (disparity / 16.).astype(np.uint8)
    return disparity_scaled
    
####################################################################    
# Project to each point in 3D
def projectTo3D(disparity):
    threeDImage = cv2.reprojectImageTo3D(disparity, Q)
    return threeDImage
         
####################################################################
                    ### RUNNING METHODS ###
####################################################################

# read in image in black and white
img1 = cv2.imread(img1,0)
img2 = cv2.imread(img2,0)

# using SIFT
keypoints1, descriptor1 = siftFeatureDetection(img1)
keypoints2, descriptor2 = siftFeatureDetection(img2)

good, pts1, pts2 = featureMatcher(img1, img2, keypoints1, keypoints2, descriptor1, descriptor2)
F, Fmask = getFundamentalMatrix(pts1, pts2)
E = getEssentialMatrix(F)
R,T, match_inliers1, match_inliers2 = find_camera_matrices_rt(E, Fmask, pts1, pts2)
img_rect1, img_rect2, Q = rectifyImages(img1, img2, R, T)
checkRectify(img_rect1, img_rect2)

#disparity = computeStereoDepth(img_rect1, img_rect2)


#threeDImage = projectTo3D(disparity)
#cv2.imshow('image', threeDImage)
#plt.show()


#pts3d = triangluation(match_inliers1, match_inliers2, R, T)
#plot3dPointCloud(pts3d)
