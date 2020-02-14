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
# image_name = 'POP0'
image_name = 'DeathStar'
img1 = cv2.imread(image_name + '1.jpg',0)  #queryimage # left image
img2 = cv2.imread(image_name + '2.jpg',0) #trainimage # right image

###### COMPUTE FUNDAMENTAL MATRIX USING HAND-MADE RANSAC

N = 15
flag_F = 0
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
	
	
	FRansac, mask = cv2.findFundamentalMat(points1[:,0:2],points2[:,0:2],cv2.FM_7POINT)
	F = FRansac[0:3,0:3]
	epilines = F.dot(points1.transpose())
	d = abs(((epilines)*(points2.transpose())).sum(axis=0))/((epilines[0]**2+epilines[1]**2)**0.5)
	
	# d_min = 2E-12
	d_min = 2E-13


	nb_inliners = 0
	for data in d:
		if data<d_min:
			nb_inliners = nb_inliners + 1
			print ("Inliner!", data, nb_inliners)
		print ("No inliner found>>>>>", data)
	print ("INLINERS:", nb_inliners_best )	
	print ("\n\n")
	if nb_inliners > nb_inliners_best:
		flag_F = 1
		nb_inliners_best = nb_inliners
		best_F = F
		best_P1 = points1
		best_P2 = points2
		d_best = d
		print ("OOOOO", best_F)
	# if nb_inliners_best >=5:
		# break
pts1 = np.float32(pts1)
pts2 = np.float32(pts2)
	
if flag_F == 0:		
	print ("INLINERS:", nb_inliners )	
	print ("DIST,norm", d/np.max(d))
	print ("RATIO:", (np.array(np.where((d/np.max(d))<0.1))))
	print ("DATA USED (regular):\n", d)
	drawFundamental(img1,img2,pts1,pts2,F)
else:
	print ("INLINERS:\n", nb_inliners_best )
	print ("DIST,norm", d_best/np.max(d_best))
	print ("RATIO", (np.array(np.where((d_best/np.max(d_best))<0.1))) )
	print ("Distances USED (best):\n", d_best )
	print ("F USED (best):\n", best_F )
	drawFundamental(img1,img2,pts1,pts2,best_F)

plt.show()
