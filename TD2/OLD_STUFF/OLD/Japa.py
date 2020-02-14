# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
inliers=0
nb_inliers_best=0

###### Read images
img1 = cv2.imread('POP01.jpg',0)  #queryimage # left image
img2 = cv2.imread('POP02.jpg',0) #trainimage # right image
# img1 = cv2.imread('DeathStar1.jpg',0)  #queryimage # left image
# img2 = cv2.imread('DeathStar2.jpg',0) #trainimage # right image


###### Detect and match keypoints
kaze = cv2.KAZE_create(upright = False,#Par défaut : false
                      threshold = 0.001,#Par défaut : 0.001
                      nOctaves = 5,#Par défaut : 4
                      nOctaveLayers = 4,#Par défaut : 4
                      diffusivity = 2)#Par défaut : 2


# find the keypoints and descriptors with KAZE
kp1, des1 = kaze.detectAndCompute(img1,None)
kp2, des2 = kaze.detectAndCompute(img2,None)

print('Nb of keypoints: ' + str(len(kp1)) + ' ' + str(len(kp2)))


# match keypoints using FLANN library
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

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
   
###### Compute Fundamental Matrix using hand-made RANSAC

for j in range (0,100):
	pts1 = []
	pts2 = []
	p1=[]
	p2=[]	
	
	# filter matching using threshold and ratio test as per Lowe's paper
	for i,(m,n) in enumerate(matches):
		if (m.distance < 0.9) & (m.distance < 0.7*n.distance): 
			pts2.append(kp2[m.trainIdx].pt)
			pts1.append(kp1[m.queryIdx].pt)
	# Choosing 7 random points
	for i in range(0,7):
		idx=random.randint(0, len(pts1)-1)
		#print(str(idx) +" "+ str(pts1[idx]))
		p1.append(pts1[idx])
		p2.append(pts2[idx])

	

	p1 = np.float32(p1)
	p2 = np.float32(p2)

	#Determining the fundamental matrix for the 7 random points
	FRansac_test, mask = cv2.findFundamentalMat(p1,p2,cv2.FM_7POINT)
	
	
	#Finding epipolaires lines
	indexes = np.random.randint(0, p1.shape[0], size=(10))
	indexes=range(p1.shape[0])
	samplePt1 = p1[indexes,:]
	samplePt2 = p2[indexes,:]
	lines1 = cv2.computeCorrespondEpilines(samplePt2.reshape(-1,1,2), 2,FRansac_test[0:3])
	lines1 = lines1.reshape(-1,3)
	lines2 = cv2.computeCorrespondEpilines(samplePt1.reshape(-1,1,2), 2,FRansac_test[0:3])
	lines2 = lines2.reshape(-1,3)


	#Finding inliners
	inliers=[]
	for lines in lines2:
		for pts in p2:
			if(abs(lines[0]*pts[0]+lines[1]*pts[1]+lines[2])< 1):
				inliers.append(pts)
				

	nb_inliers=len(inliers)
	
	#Determining if new inliers are better than the last
	if(nb_inliers>nb_inliers_best):
		nb_inliers_best= nb_inliers
		FRansac=FRansac_test[0:3]
		inliers_best=inliers
		p1_best=p1
		p2_best=p2
		print('Number of RANSAC inliers : ' + str(nb_inliers_best))

	
#Drawing the best inliers
drawFundamental(img1,img2,p1_best, p2_best,FRansac)


plt.show()
