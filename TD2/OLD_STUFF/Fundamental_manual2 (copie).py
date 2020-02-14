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
	F_7 = F_7[:3]	#In case there are multiple solutions
	# if (shape(F_7)[0]==9):	#In case there are multiple solutions, randomly pick one
		# index_F = np.array(random.sample(range(3),1))
		# F_7 = F_7[int(index_F*3):int(index_F*3+3),:]	
		
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
			
			if d<Ransac_seuil:
				if i not in inliner_list:	## Éviter redondance de points
					nb_inliners += 1
					inliner_pts1.append(pts1[i])
					inliner_pts2.append(pts2[i])
					inliner_list.append(i)
			i += 1
	
		# print ("Inliners: ", nb_inliners)
		# print ("---")
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

print('Nb of keypoints: ' + str(len(kp1)) + ' ' + str(len(kp2)))
#imgd=img1
#imgd = cv2.drawKeypoints(img1, kp1, imgd,-1,flags=4)
#cv2.imshow('Keypoints', imgd)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


# match keypoints using FLANN library
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

#m_image = np.array([])
#m_image = cv2.drawMatches(
#    img1, kp1,
#    img2, kp2,
#    [match[0] for match in matches],
#    m_image)
#cv2.imshow('Match', m_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

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
N = 1
nb_inliners_best= 0
nb_inliners_best2= 0
bestF_7 = np.zeros((3,3))
for SampleSize in range(N):
	
	print ("             ",SampleSize, "<<<<")
	print ("FIRST RANSAC")
	
	F_7, nb_inliners, pts1_7, pts2_7, inliner_pts1, inliner_pts2 = Run_7points (pts1, pts2, 1.5)  #POP = 50, STAR = 
	

	## Vérifie à nouveau quelle est la meilleure matrice fondamentale par rapport au nombre d'inliners
	print ("Best and atual (F): ",nb_inliners_best, nb_inliners)
	if nb_inliners > nb_inliners_best:
		## if nb_inliners_best/pts1.shape[0] < 0.7:
		x_prime = []
		x_orig = []
		add = []
		for i in range(shape(inliner_pts1)[0]):
			inliner_pts1 = np.array(inliner_pts1)
			inliner_pts2 = np.array(inliner_pts2)
			x_orig.append([inliner_pts1[i,0], inliner_pts1[i,1],1])
			x_prime.append([inliner_pts2[i,0], inliner_pts2[i,1],1])
			# print (inliner_pts2)
		for j in range (shape(inliner_pts1)[0]):
			print (np.array(x_orig), np.array(x_orig[j, :]))
			# a = np.dot(x_orig, F_7)
			# add.append()
		print ("++++++++++++++XFX tests: ", shape(x_orig), shape(F_7), shape(np.array(x_prime).transpose()))
		
		print ("++++++", shape(a))
		print ("+++++++++++", shape(np.dot(a, np.array(x_prime).transpose())))
		print ("Raio: ", nb_inliners/shape(pts1)[0])
		print ("Norma: ", LA.norm(bestF_7-F_7))
		nb_inliners_best = nb_inliners
		bestF_7 = F_7
		best_pt1 = pts1_7
		best_pt2 = pts2_7
		print ("                BEST1 F")
		best_line=SampleSize
	
	
print("                         +++MELHOR", best_line)
print("Porcentagem:", nb_inliners_best/pts1.shape[0])
print ("IMAGE SIZE: ", shape(img1))
# drawFundamental(img1,img2,best_pt1,best_pt2,bestF_7)		
	

plt.show()
