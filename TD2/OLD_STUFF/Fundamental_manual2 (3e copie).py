import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import shape
import random
import math
from numpy import linalg as LA


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
  
  
  
def Run_7points (pts1, pts2, Ransac_seuil):
	## Gets list of random items of length 7 from the given list. 
	index_7 = random.sample(range(pts1.shape[0]), 7)
	pts1_7 = pts1[index_7]
	pts2_7 = pts2[index_7]
	
	## Matrice fondamentale à partir de l’algorithme de 7 points
	F_7, mask = cv2.findFundamentalMat(pts1_7,pts2_7,cv2.FM_7POINT,1)
	F_7 = F_7[:3]	#In case there are multiple solutions, randomly pick one
	
	## Find epilines corresponding to points in left image (first image) and drawing its lines on right image
	epilines = cv2.computeCorrespondEpilines(pts1_7.reshape(-1,1,2), 1, F_7)
	epilines = epilines.reshape(-1,3)  

	## Calcule la distance entre les pts2 de l'image et les lignes épipolaires
	nb_inliners = 0
	inliner_pts1=[]
	inliner_pts2=[]
	inliner_list=[]
	
	for line in epilines:	## Pour chaque ligne épipolaire
		i=0
		for point2 in pts2: ## Pour chaque point pts2 de l'image		
			## Vérifier la quantité de inliners e les sauvegarder
			d = abs(np.dot(line[0:2],point2)+line[2])/(line[1]**2+line[0]**2)**0.5
			# print ("distance: ", d)
			if d < Ransac_seuil:
				if i not in inliner_list:	## Éviter des redondances des points
					nb_inliners += 1
					inliner_pts1.append(pts1[i])
					inliner_pts2.append(pts2[i])
					inliner_list.append(i)
			i += 1
	return F_7, nb_inliners, pts1_7, pts2_7, inliner_pts1, inliner_pts2
	
	
###### READ IMAGES
image_name = 'POP0'
# image_name = 'DeathStar'
img1 = cv2.imread(image_name + '1.jpg',0)  #queryimage # left image
img2 = cv2.imread(image_name + '2.jpg',0) #trainimage # right image

###### Detect and match keypoints
kaze = cv2.KAZE_create(upright = False,#Par défaut : false
                      threshold = 0.001,#Par défaut : 0.001
                      nOctaves = 4,#Par défaut : 4
                      nOctaveLayers = 4,#Par défaut : 4
                      diffusivity = 2)#Par défaut : 2


# find the keypoints and descriptors with KAZE
kp1, des1 = kaze.detectAndCompute(img1,None)
kp2, des2 = kaze.detectAndCompute(img2,None)

# match keypoints using FLANN library
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
pts1 = []
pts2 = []

# filter matching using threshold and ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if (m.distance < 0.9) & (m.distance < ((2)**0.5/2)*n.distance):  #Par défaut : 0.99. For Lowe's paper, 0.7
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.float32(pts1)
pts2 = np.float32(pts2)

print('Number of matched points : ' + str(pts1.shape[0]))

   
   
###### COMPUTE FUNDAMENTAL MATRIX USING HAND-MADE RANSAC
N = 600
nb_inliners_best= 0
nb_inliners_best2= 0
ratio_err = 0
err_best = 1000
for SampleSize in range(N):
	inliners_N = 0
	## Éxecute l'algorithme des 7-points
	F_7, nb_inliners, pts1_7, pts2_7, inliner_pts1, inliner_pts2 = Run_7points (pts1, pts2, 2)  #POP = 50, STAR = 
	
	## Éxecute l'algorithme des 7-points avec les inliners
	inliner_pts1_2 = np.array(inliner_pts1)
	inliner_pts2_2 = np.array(inliner_pts2)
	F_7, nb_inliners2, pts1_7, pts2_7, inliner_pts1, inliner_pts2 = Run_7points (inliner_pts1_2, inliner_pts2_2, 0.5)  #POP = 50, STAR = 
	
	inliners_N = nb_inliners + nb_inliners2
	## Vérifie à nouveau quelle est la meilleure matrice fondamentale par rapport au nombre d'inliners
	# inliners_N = inliners_N + nb_inliners

	# print ("                                      Inliners values: ", nb_inliners_best, nb_inliners)
	if inliners_N > nb_inliners_best:
		# print ("<<<<<<<<<<<<<<<<<<<", SampleSize)
		# print ("INLINERS: ", inliners_N)
		# print ("Inliners first ransac: ", shape(pts1)[0], nb_inliners, nb_inliners/shape(pts1)[0])
		ratio = nb_inliners2/shape(inliner_pts1_2)[0]
		# print ("Inliners second ransac: ", shape(inliner_pts1_2)[0], nb_inliners2, nb_inliners2/shape(inliner_pts1_2)[0])			
		
		# if ratio > ratio_err and ratio < 0.71:
			# ratio_err = ratio
		nb_inliners_best = inliners_N
		bestF_7 = F_7
		best_pt1 = pts1_7
		best_pt2 = pts2_7
		best_line=SampleSize
		print ("                BEST F. Inliners: ", nb_inliners_best)
print ("++++++++ BEST LINE: ", best_line)
drawFundamental(img1,img2,best_pt1,best_pt2,bestF_7)		
	

plt.show()
