import numpy as np
import cv2
from numpy import shape
from matplotlib import pyplot as plt

roi_defined = False
 
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

# cap = cv2.VideoCapture('Sequences/Antoine_Mug.mp4')
# cap = cv2.VideoCapture('Sequences/VOT-Ball.mp4')
# cap = cv2.VideoCapture('Sequences/VOT-Basket.mp4')
# cap = cv2.VideoCapture('Sequences/VOT-Car.mp4')
# cap = cv2.VideoCapture('Sequences/VOT-Sunshade.mp4')
cap = cv2.VideoCapture('Sequences/VOT-Woman.mp4')


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

roi = frame[c:c+w,r:r+h]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Pixels with S<30, V<20 or V>235 are ignored
mask = cv2.inRange(hsv_roi, np.array((0.,30.,20.)), np.array((180.,255.,235.)))
# mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ) # 10 = iterations maximales, 1 = movement maximal
cpt = 1

while(1):
	ret ,frame = cap.read()
	if ret == True:

		#### ORIENTATION LOCALE
		img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		grad_x = cv2.Sobel(img,cv2.CV_32F,1,0)  # x
		grad_y = cv2.Sobel(img,cv2.CV_32F,0,1)  # y
		gradient = np.hypot(grad_x, grad_y)/256
		img2 = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
		img2[np.where(((img2[:,:,:]<0.75)).all(axis=2))] = [0,0,255]
		cv2.imshow('Module Gradient',img2)

		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
		x,y,h,w = track_window

		ret, track_window = cv2.meanShift(dst, track_window, term_crit)
		cv2.imshow('Sequence',frame)
				
		k = cv2.waitKey(60) & 0xff
		if k == 27:
			break
		elif k == ord('s'):
			cv2.imwrite('Frame_%04d.png'%cpt,frame_tracked)
		cpt += 1
	else:
		break

cv2.destroyAllWindows()
cap.release()

