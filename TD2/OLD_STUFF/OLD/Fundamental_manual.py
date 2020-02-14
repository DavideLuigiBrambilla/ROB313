import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import shape
import random
import math


##### Definition of some helper functions
def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color=tuple(cv2.cvtColor(np.asarray([[[np.random.randint(0,180),255,255]]],dtype=np.uint8),cv2.COLOR_HSV2BGR)[0,0,:].tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,2)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def drawFundamental(img1,img2,pts1,pts2,F):
    # Find epilines corresponding to some points in right image (second image) and
    # drawing its lines on left image
    indexes = np.random.randint(0, pts1.shape[0], size=(10))
    indexes=range(pts1.shape[0])
    samplePt1 = pts1[indexes,:]
    samplePt2 = pts2[indexes,:]

    lines1 = cv2.computeCorrespondEpilines(samplePt2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,samplePt1,samplePt2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(samplePt1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,samplePt2,samplePt1)

    plt.figure(figsize=(15, 5))
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    
###### READ IMAGES
image_name = 'POP0'
# image_name = 'DeathStar'
img1 = cv2.imread(image_name + '1.jpg',0)  #queryimage # left image
img2 = cv2.imread(image_name + '2.jpg',0) #trainimage # right image

###### COMPUTE FUNDAMENTAL MATRIX USING HAND-MADE RANSAC

N = 1

nb_inliners_best= 0
for SampleSize in range(N):
# while(1):
	###### DETECT THE KEYPOINTS
	kaze = cv2.KAZE_create(upright = False,		#Par défaut : false
						  threshold = 0.001,	#Par défaut : 0.001 / 0.001
						  nOctaves = 4,			#Par défaut :     4 / 6
						  nOctaveLayers = 4,	#Par défaut :     4 / 6
						  diffusivity = 2)		#Par défaut :     2 / 4

	## Find the keypoints and descriptors with KAZE
	kp1, des1 = kaze.detectAndCompute(img1,None)
	kp2, des2 = kaze.detectAndCompute(img2,None)
	
	## Match keypoints using FLANN library
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)
	flann = cv2.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)
	
	pts1 = []
	pts2 = []
			
	## Filter matching using threshold and ratio test as per Lowe's paper
	for i,(m,n) in enumerate(matches):
		if (m.distance < 0.9) & (m.distance < 0.7*n.distance):  #Par défaut : 0.99. For Lowe's paper, 0.7
			pts2.append(kp2[m.trainIdx].pt)
			pts1.append(kp1[m.queryIdx].pt)
			
	## Gets 7 random matches
	points1 = []
	points2 = []

	for i in range (7):
		rand_number = random.randrange(0, shape(pts1)[0])
		points1.append([np.array(pts1)[rand_number,0], np.array(pts1)[rand_number,1],1])
		points2.append([np.array(pts2)[rand_number,0], np.array(pts2)[rand_number,1],1])
	
	points1 = np.float32(points1)
	points2 = np.float32(points2)
	
	# img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
	# img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
	# color=tuple(cv2.cvtColor(np.asarray([[[np.random.randint(0,180),255,255]]],dtype=np.uint8),cv2.COLOR_HSV2BGR)[0,0,:].tolist())
	# img1 = cv2.circle(img1,tuple(points1[0]),35,color,-1)
	# img2 = cv2.circle(img2,tuple(points2[0]),35,color,-1)
	# plt.subplot(121),plt.imshow(img1)
	# plt.subplot(122),plt.imshow(img2)	
	
	
	
	FRansac, mask = cv2.findFundamentalMat(points1[:,0:2],points2[:,0:2],cv2.FM_7POINT)
	F = FRansac[0:3,0:3]
	epilines = F.dot(points1.transpose())
	d = abs(((epilines)*(points2.transpose())).sum(axis=0))/((epilines[0]**2+epilines[1]**2)**0.5)

	
	nb_inliners = 0
	
	for index_line in range(7):
		# ## Calculates the distance
		a = lines2[index_line][0]
		b = lines2[index_line][1] 
		c = lines2[index_line][2]
		x = points2[index_line][0]
		y = points2[index_line][1]
		
		d = abs(a*x + b*y + c)/(a**2 + b**2)**0.5
		if d==0:
			nb_inliners = nb_inliners + 1
		print ("Distance", d, nb_inliners)
	if nb_inliners > nb_inliners_best:
		nb_inliners_best = nb_inliners
		best_F = F
		print ("OOOOO", best_F)
		if nb_inliners_best >=6:
			break
	
	
	
	# --------------------------------------------------------------------------------------------------------
	# FRansac, mask = cv2.findFundamentalMat(points1,points2,cv2.FM_7POINT)
	
	
	# F = FRansac[0:3,0:3]
	
	
	## Find epilines corresponding to points in left image (first image) and
	## drawing its lines on right image
	# lines2 = cv2.computeCorrespondEpilines(points1.reshape(-1,1,2), 1,F)
	# lines2 = lines2.reshape(-1,3)
	# img3,img4 = drawlines(img2,img1,lines2,points2,points1)
	# print("Shape line", shape(lines2), points2[0], points2[0][0], points2[0][1] )
	
	# nb_inliners = 0
	
	# for index_line in range(7):
		## Calculates the distance
		# a = lines2[index_line][0]
		# b = lines2[index_line][1] 
		# c = lines2[index_line][2]
		# x = points2[index_line][0]
		# y = points2[index_line][1]
		
		# d = abs(a*x + b*y + c)/(a**2 + b**2)**0.5
		# if d==0:
			# nb_inliners = nb_inliners + 1
		# print ("Distance", d, nb_inliners)
	# if nb_inliners > nb_inliners_best:
		# nb_inliners_best = nb_inliners
		# best_F = F
		# print ("OOOOO", best_F)
		# if nb_inliners_best >=6:
			# break
	
	# -------------------------------------------------------------------------------
	# plt.subplot(121),plt.imshow(img5)
	# plt.subplot(122),plt.imshow(img3)
	# plt.show()
	
	# print ("Shape", shape(points1), shape(points2), shape(FRansac))
	
	# print("Data", F, "\n\n", points1, "\n\n", points2 )
	
	## select inlier points
	# inlierpts1 = point1[mask.ravel()==1]
	# inlierpts2 = point2[mask.ravel()==1]
	# print("Loop", SampleSize, "/", N)
	## plot epipolar lines
# drawFundamental(img1,img2,points1,points2,best_F)



plt.show()
