#####################################################################

# Structure from motion: Computer Vision Assignment L4

# Author : Christian Johnston, christian.johnston@durham.ac.uk

# OpenCV 3.4.2

# Credit for code-sections to:
# Michael Beyeler
# Toby Breckon Durham University

# NOTE- For visualisation, Open3D was used to plot the 3D point cloud.

# To use SIFT/SURF may need to run the command: pip install opencv-contrib-python==3.4.2.16

#####################################################################

import cv2
import os
import numpy as np
import random
import csv
import math
import matplotlib.pyplot as plt
import scipy.spatial as spatial
from mpl_toolkits.mplot3d import Axes3D
import open3d as op


#####################################################################
# Directory to cycle, please change as appropriate 
directory_to_cycle = "./OH-sfm-test-sequence-DURHAM-2015-03-27-calibrated/2015-03-27_10-47-08_Seq2/monocular_left_calibrated"

####################################################################

def siftFeatureDetection(img):
    sift = cv2.xfeatures2d.SIFT_create()
    keyPoints, descriptor = sift.detectAndCompute(img, None)
    return keyPoints, descriptor

####################################################################

# SURF feature point detection
def surfFeatureDetection(img):
    surf = cv2.xfeatures2d.SURF_create(350)
    keyPoints, descriptor = surf.detectAndCompute(img, None)
    return keyPoints, descriptor

####################################################################

def plotFeaturePoints(img, keyPoints):
    img = cv2.drawKeypoints(img, keyPoints, img)
    cv2.imshow('image', img)
    plt.show()

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
    matchedFeaturesImage = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, good2, None, flags = 2)
    #cv2.imshow('match', matchedFeaturesImage)
    return good, pts1, pts2

####################################################################
# FLANN based matcher
def flannMatcher(keypoints1,keypoints2, descriptor1, descriptor2):

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    knn_matches = flann.knnMatch(descriptor1, descriptor2, k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(knn_matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts1.append(keypoints1[m.queryIdx].pt)
            pts2.append(keypoints2[m.trainIdx].pt)

    return good, pts1, pts2

####################################################################
# pts1 and pts2 from ratio test as above
def getFundamentalMatrix(pts1, pts2):
    # convert to numpy
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.5, 0.99)
    return F, mask

####################################################################
# Eessential matrix = K^ . F . K
def getEssentialMatrix(F):
    # K is the intrinsic camera paramaters
    k = np.array([[1.9304844e+03, 0, 6.14632329e+02], [0, 1.18642523e+03, 4.33180837e+02], [0, 0, 1]])
    
    # F is the fundamental matrix (T does transpose)
    E = k.T.dot(F).dot(k)
    return E

####################################################################
# Draw epipolar lines 
def draw_epipolar_lines(img1, img2, match_pts1, match_pts2, F):
        match_pts1 = np.int32(match_pts1)
        match_pts2 = np.int32(match_pts2)

        lines1 = cv2.computeCorrespondEpilines(match_pts2.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)
        img3, img4 = draw_epipolar_lines_helper(img1, img2, lines1, match_pts1, match_pts2)

        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(match_pts1.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)
        img1, img2 = draw_epipolar_lines_helper(img2, img1, lines2, match_pts2, match_pts1)
        cv2.imshow("left", img1)
        cv2.imshow("right", img2)

####################################################################
def draw_epipolar_lines_helper(img1, img2, lines, pts1, pts2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    c = img1.shape[1]
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0]*c) / r[1]])
        cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        cv2.circle(img1, tuple(pt1), 5, color, -1)
        cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


####################################################################
def find_camera_matrices_rt(E, Fmask, pts1, pts2):
    first_inliers = []
    second_inliers = []
    K = np.array([[1.9304844e+03, 0, 6.14632329e+02], [0, 1.18642523e+03, 4.33180837e+02], [0, 0, 1]])
    K_inv = np.linalg.inv(K)
    for i in range(len(Fmask)):
        if Fmask[i]:
            # normalize and homogenize the image coordinates
            first_inliers.append(K_inv.dot([pts1[i][0], pts1[i][1], 1.0]))
            second_inliers.append(K_inv.dot([pts2[i][0],pts2[i][1], 1.0]))
     
    '''   
    R1, R2, T = cv2.decomposeEssentialMat(E)

    finalR = R1
    if(R1[1][1] < 0.9 or R1[1][1] > 1.1):
        finalR = R2
    '''

    _, finalR, T, _ = cv2.recoverPose(E, np.array(pts1), np.array(pts2), K)

    return finalR, -T, first_inliers, second_inliers


####################################################################
def triangluation(match_inliers1, match_inliers2, R, T):

    # first camera is no translation or rotation
    # Second camera consists of R, T
    global oldR
    global oldT
   
    Rt1 = np.hstack((oldR, oldT))

    # get into vector form
    oldR = cv2.Rodrigues(oldR)[0]
    R = cv2.Rodrigues(R)[0]

    # compose new R and T (composition of old ones)
    newR, newT, _, _, _, _,_,_,_,_ = cv2.composeRT(oldR, oldT, R, T)

    #transform back to matrix format
    newR = cv2.Rodrigues(newR)[0]
    #print(newR)
    print(newT)

    # Bring R and T together
    Rt2 = np.hstack((newR, newT))
    
    first_inliers = np.array(match_inliers1).reshape(-1,3)[:, :2]
    second_inliers = np.array(match_inliers2).reshape(-1,3)[:, :2]
    
    # This returns the trianglulated real-world points using 4D homogeneous coordinates
    pts4D = cv2.triangulatePoints(Rt1, Rt2, first_inliers.T, second_inliers.T).T
    
    oldR = newR
    oldT = newT
    
    return pts4D


####################################################################
def plot3dPointCloud(pts3D):
    Ys = pts3D[:,0]
    Zs = pts3D[:,1]
    Xs = pts3D[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(Xs, Ys, Zs, c='r', s = 3, marker='o')
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_zlabel('X')
    plt.title('3D point cloud: Use pan axes button below to inspect')
    plt.show()
    return

####################################################################
def display(pts3D):
    pcd = op.PointCloud()
    pcd.points = op.Vector3dVector(pts3D)
    op.draw_geometries([pcd])
    return

####################################################################
# Rectify the images
# Taken from Toby Breckon GitHub: python-examples-cv/stereo_sgbm.py 
def rectifyImages(img1, img2, R, T):
    # d and K found from files on DUO
    K = np.array([[1.9304844e+03, 0, 6.14632329e+02], [0, 1.18642523e+03, 4.33180837e+02], [0, 0, 1]])
    d = np.array([-4.8441890104482732e-001, 3.1770182182461387e-001, 4.8167296939537890e-003, 5.9334794668205733e-004,
-1.4902486951308128e-001]).reshape(1,5)

    # alpha as -1 seems to give me the correct Q
    RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(K, d, K, d, img1.shape[:2], R, T, alpha=-1)

    mapL1, mapL2 = cv2.initUndistortRectifyMap(K, d, RL, K, img1.shape[:2], cv2.CV_32FC1);
    mapR1, mapR2 = cv2.initUndistortRectifyMap(K, d, RR, K, img2.shape[:2], cv2.CV_32FC1);

    img_rect_1 = cv2.remap(img1, mapL1, mapL2, cv2.INTER_LINEAR);
    img_rect_2 = cv2.remap(img2, mapR1, mapR2, cv2.INTER_LINEAR);
    return img_rect_1, img_rect_2, Q


#################################################################### 
def computeDisparity(img_rect1, img_rect2):
    max_disparity = 128
    min_disparity = 0
    num_disparities = max_disparity - min_disparity # divisible by 16
    window_size = 21 # odd number in 3--11 range
    stereoProcessor = cv2.StereoSGBM_create(min_disparity, num_disparities, window_size);
    disparity = stereoProcessor.compute(img_rect1,img_rect2)

    cv2.filterSpeckles(disparity, 0, 400, max_disparity-5)
    _, disparity = cv2.threshold(disparity,0, max_disparity*16, cv2.THRESH_TOZERO)
    disparity_scaled = (disparity / 16).astype(np.uint8)
    return disparity_scaled                                            
        
####################################################################    
# Project to each point in 3D and save these in a file out.ply
def projectTo3D(disparity, Q):
    points = cv2.reprojectImageTo3D(disparity, Q)
    return points

####################################################################
def writeToFile(img1, disparity, Q):
    colors = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    mask = disparity > disparity.min()
    out_points = points[mask]
    out_colours = colors[mask]
    out_fn = 'out.ply'
    write_ply('out.ply', out_points, out_colors)

####################################################################
# function to compute the average position of the 3D points and remove outliers
def avg3DPoints(threeDPoints):
    # Arbitrary threshold
    threshold = 15
    # total = [0,0,0]
    threeDPointsTot = np.zeros((1,3))
    # loop through and add points
    for point in threeDPoints:
        threeDPointsTot += point
    # compute average
    threeDPointsAvg = threeDPointsTot / len(threeDPoints)

    # compute distance to average
    for point in range(len(threeDPoints)-1, 0, -1):
        distance = np.linalg.norm(threeDPoints[point] - threeDPointsAvg)
        # if greater than threshold
        if(distance > threshold):
            #remove from numpy array
            threeDPoints = np.delete(threeDPoints, point, 0)
    return threeDPoints
        

####################################################################
                    ### RUNNING METHODS ###
####################################################################

def runAll(img1, img2, counter):
    # read in image in black and white
    img1 = cv2.imread(img1,0)
    img2 = cv2.imread(img2,0)

    # SURF feature detection
    keypoints1, descriptor1 = surfFeatureDetection(img1)
    keypoints2, descriptor2 = surfFeatureDetection(img2)
    
    #good, pts1, pts2 = featureMatcher(img1, img2, keypoints1, keypoints2, descriptor1, descriptor2)
    good, pts1, pts2 = flannMatcher(keypoints1,keypoints2, descriptor1, descriptor2)

 
    # Looping to find good Z value
    cnter = 0

    while True:
        if cnter > 20:
            break;
        F, Fmask = getFundamentalMatrix(pts1, pts2)
        E = getEssentialMatrix(F)
        
        # Get the camera matrices 
        R,T,match_inliers1, match_inliers2 = find_camera_matrices_rt(E, Fmask, pts1, pts2)

        # Loop untill you get a small z-value
        zValue = T[2][0]
        if(abs(zValue) < 0.1):
            break;

        # shuffle list of points (as fundamental matrix calculaiton is deterministic)
        shuffleList = list(range(0, len(pts1)))
        random.shuffle(shuffleList)
        pts1 = [pts1[i] for i in shuffleList]
        pts2 = [pts2[i] for i in shuffleList]
        cnter+=1

    # Rectify images
    img_rect1, img_rect2, Q = rectifyImages(img1, img2, R, -T)

    # Calculate Disparity
    disparity = computeDisparity(img_rect1, img_rect2)
    
    # Reproject back to 3D
    points = projectTo3D(disparity, Q)

    # triangulation
    pts4D = triangluation(match_inliers1, match_inliers2, R, T)

    # Global point cloud variables
    global total3DPointCloud

    # convert 4D points to 3D points
    pts3D = pts4D[:, :3]/np.repeat(pts4D[:,3], 3).reshape(-1,3)

    # remove outliers
    pts3D = avg3DPoints(pts3D)

    # Display the individual 3D point cloud
    #display(pts3D)

    # append to gloval point cloud 
    total3DPointCloud = np.append(total3DPointCloud, pts3D, axis=0)

    #Display the total 3D point cluster after a given number of frames
    #if(counter == 30):
        #display(total3DPointCloud)


####################################################################
# Global Variables

# Global 4D point cloud
total3DPointCloud = np.empty([0,3])

# Original R and T
oldR = np.eye(3, 3)
oldT = np.zeros((3, 1))

#####################################################################
# Loop through all images in the sequence 

firstPass = True
counter = 1;
prevImage = None
currentImage = None

for filename in sorted(os.listdir(directory_to_cycle)):
    if '.png' in filename:
        currentImage= os.path.join(directory_to_cycle, filename)
        if firstPass:
            firstPass = False
        else:
            counter+=1
            # Run Structure from motion
            runAll(prevImage, currentImage, counter)
            key = cv2.waitKey(200) # wait 200ms
            if (key == ord('x')):
                break
        prevImage = currentImage

display(total3DPointCloud)
cv2.destroyAllWindows()


