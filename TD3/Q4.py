import matplotlib.pyplot as plt
import numpy as np
from numpy import shape
import cv2
	
roi_defined = False
gradient_threshold = 0.5

def define_ROI(event, x, y, flags, param):
	global r,c,w,h,roi_defined
	if event == cv2.EVENT_LBUTTONDOWN:
		r, c = x, y
		roi_defined = False
	elif event == cv2.EVENT_LBUTTONUP:
		r2, c2 = x, y
		h = abs(r2-r)
		w = abs(c2-c)
		r = min(r,r2)
		c = min(c,c2)  
		roi_defined = True

#### DEFINE VIDEO SEQUENCE #########
# cap = cv2.VideoCapture('Sequences/Antoine_Mug.mp4')
# cap = cv2.VideoCapture('Sequences/VOT-Ball.mp4')
# cap = cv2.VideoCapture('Sequences/VOT-Basket.mp4')
cap = cv2.VideoCapture('Sequences/VOT-Car.mp4')
# cap = cv2.VideoCapture('Sequences/VOT-Sunshade.mp4')
# cap = cv2.VideoCapture('Sequences/VOT-Woman.mp4')

#### DEFINE ROI
ret,frame = cap.read()
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)

while True:
	cv2.imshow("First image", frame)
	key = cv2.waitKey(1) & 0xFF
	if (roi_defined):
		cv2.rectangle(frame, (r, c), (r+h, c+w), (0, 255, 0), 2)
	else:
		frame = clone.copy()
	if key == ord("q"):
		break
track_window = (r,c,h,w) 
roi = clone[c:c+w,r:r+h]

#### MODÈLE IMPLICITE
roi_hough = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
grad_x = cv2.Sobel(roi_hough,cv2.CV_32F,1,0)  # x
grad_y = cv2.Sobel(roi_hough,cv2.CV_32F,0,1)  # y
roi_hough = np.hypot(grad_x, grad_y)/256
roi_hough[roi_hough[:,:] < gradient_threshold] = 0 #Defined at line 7
cv2.imshow('Model',roi_hough)
         
#### CREATES THE R-TABLE
# Creates the list for the R-table
table = [[0 for j in range(1)] for i in range(90)]

# Computes the center of the image (reference)
center_img = [int(shape(roi_hough)[0]/2), int(shape(roi_hough)[1]/2)]

# Computes the paramters of the model
for x1 in range(shape(roi_hough)[0]):
	for y1 in range(shape(roi_hough)[1]):
		if (roi_hough[x1, y1] != 0):	# If the point is in the border of the model
			# Point of referencce
			x2 = center_img[0]
			y2 = center_img[1]
			
			# finds the paramters for each angle
			r = [(x2-x1), (y2-y1)]
			if (x2-x1 != 0):
				# Finds the value of theta
				theta = int(np.rad2deg(np.arctan(int((y2-y1)/(x2-x1)))))
			else:
				theta = 0
				r = 0
			if (r != 0):
				# Adds the value of r to the R-table at the position of the angle
				table[np.absolute(theta)].append(r)
for i in range(len(table)):
	# After completing the table, eliminates the null values
	table[i].pop(0)
cpt = 0
while(1):
	ret ,frame = cap.read()
	if ret == True:
		
		#### ORIENTATION LOCALE
		img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		grad_x = cv2.Sobel(img,cv2.CV_32F,1,0)  # x
		grad_y = cv2.Sobel(img,cv2.CV_32F,0,1)  # y
		gradient = np.hypot(grad_x, grad_y)/256
		gradient[gradient[:,:]<gradient_threshold] = 0
		
		# Draws the tracking window
		x,y,h,w = track_window

		# Comparing with the R-table
		accumulator = np.zeros((shape(gradient)[0]+int(shape(gradient)[0]*0.2), shape(gradient)[1] +int(shape(gradient)[0]*0.2)))

		for x_comp in range(1, shape(gradient)[0]):
			for y_comp in range(shape(gradient)[1]):
				if gradient[x_comp, y_comp] != 0:  
					if (x_comp != 0):
						theta = int(np.rad2deg(np.arctan(int(y_comp/x_comp))))
					else:
						theta = 0
					orientations = table[theta]
					for count_ori in orientations:
						accumulator[count_ori[0]+x_comp, count_ori[1]+y_comp] += 1
		

		## Finds the maximum value of the Hough Transform
		x_acc, y_acc = np.unravel_index(np.argmax(accumulator), shape(accumulator))
		frame_tracked = cv2.rectangle(frame, (y_acc-int(h/2),x_acc-int(w/2)), (y_acc + int(h/2),x_acc + int(w/2)), (255,0,0) ,2)
		cv2.imshow('Sequence',frame_tracked)

		# Plots the Hough transoform
		plt.figure("Hough")
		plt.imshow(accumulator)
		plt.draw()
		plt.pause(0.0001)
		

		cpt += 1		
		k = cv2.waitKey(60) & 0xff
		if k == 27:
			break
		elif k == ord('s'):
			cv2.imwrite('Frame_%04d.png'%cpt,frame_tracked)
	else:
		break
