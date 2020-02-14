import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import shape
import random
import math


# F = [[ 1.17587202e-06, -1.96003128e-07, -1.58861335e-03], [-3.27784929e-07, -3.21263493e-07,  8.22143820e-04], [-3.20566567e-04, -5.08392027e-04,  1.00000000e+00]] 

points1 =   [[1129.0629,   837.4924,1 ],
 [1142.6692 ,  916.9587,1 ],
 [ 593.078 ,  1134.0383,1 ],
 [1148.4059  , 857.1172 ,1],
 [1451.7183 , 1045.2439 ,1],
 [ 595.5438  , 987.5935,1 ],
 [ 621.91125 ,1043.2689,1 ]] 


points2 =  [[ 760.68506,  823.31396,1],
 [ 688.374  ,  868.55743,1],
 [ 361.81717 , 626.85925,1],
 [ 746.1253 ,  843.4132,1 ],
 [ 703.0143 , 1185.8911,1 ],
 [1182.5383 , 1117.9066,1 ],
 [ 413.9403 ,  602.3523 ,1]]


N = 1

nb_inliners_best= 0
for SampleSize in range(N):
	points1 = np.float32(points1)
	points2 = np.float32(points2)

	FRansac, mask = cv2.findFundamentalMat(points1[:,0:2],points2[:,0:2],cv2.FM_7POINT)
	F = FRansac[0:3,0:3]

	epilines = F.dot(points1.transpose())

	# print ("Shapes", shape(FRansac), shape(points1), shape(points2), shape(epilines))
	# print ("Shape", shape(epilines), shape(points2.transpose()))
	# print (epilines*points2.transpose())
	# a = epilines*points2.transpose()
	# A = epilines
	# B = points2.transpose()
	# d_1 = (A * B).sum(axis=0)
	# d_2 = epilines[0]**2+epilines[1]**2
	d = abs(((epilines)*(points2.transpose())).sum(axis=0))/((epilines[0]**2+epilines[1]**2)**0.5)

	d_min = 3e-10

	nb_inliners = 0
	for data in d:
		if data<d_min:
			nb_inliners = nb_inliners + 1
		print ("Distance", data, nb_inliners)
	if nb_inliners > nb_inliners_best:
		nb_inliners_best = nb_inliners
		best_F = F
		print ("OOOOO", best_F)
		# if nb_inliners_best >=6:
			# break
			
			
			
	# print ("oi", A[0][0]*B[0][0] + A[1][0]*B[1][0] + A[2][0]*B[2][0])
	print ("oi", d, np.min(d), np.max(d), np.mean(d))
	# np.dot(epilines[:,:], points2.transpose()[:,:]))
