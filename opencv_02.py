import cv2
import numpy as np

# 이미지를 불러오기
image = cv2.imread('./data/lena.png',0)
# 0~255를 0~1로 변경. 이미지를 부동소숫점으로 변경
image = image.astype(np.float32) / 255
print('Shape : ', image.shape)
print('Data type : ', image.dtype)

# 블루채널과 레드채널을 서로 변경
# image[:,:,[0,2]] = image[:,:,[2,0]]
# cv2.imshow('image', np.clip(image*2,0,1))
# cv2.waitKey()
# cv2.destroyAllWindows()

# 이미지 색공간 변경
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('image', gray)
# cv2.waitKey()
# cv2.destroyAllWindows()

# HSV 색공간으로 변경
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# cv2.imshow('image', hsv)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 감마보정과 요소 단위의 수학
gamma = 0.5
corrected_image = np.power(image,gamma)

cv2.imshow('image', image)
cv2.imshow('image', corrected_image)
cv2.waitKey()
cv2.destroyAllWindows()