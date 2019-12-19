# 행렬, 색상, 필터
# 행렬 조작 : 생성하기, 채우기, 요소 접근하기, ROI

# install py-opencv

import cv2, numpy as np
#                세로, 가로    
image = np.full((480,640,3),255, np.uint8)
cv2.imshow('white', image)
cv2.waitKey()
cv2.destroyAllWindows()


#                세로,가로,f  B  G  R 
image = np.full((480,640,3),(0,0,255), np.uint8)
cv2.imshow('red', image)
cv2.waitKey()
cv2.destroyAllWindows()

image[240,160] = image[240,320] = image[240,480] = (255,255,255)
cv2.imshow('black with white pixels', image)
cv2.waitKey()
cv2.destroyAllWindows()
