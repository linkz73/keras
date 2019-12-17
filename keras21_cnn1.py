from keras.models import Sequential

filter_size = 32
kernel_size = (3,3)

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
model = Sequential()
# conv2D : 이미지, conv3D : 3차원
# conv2D 7 : 컨벌루션 필터 수(아웃풋), 컨벌루션 커널 행렬(자르는 단위), padding(경계 처리 방법) (same-입력이미지 사이즈와 동일, valid-유효영역만 출력), input shape(픽셀 가로, 픽셀 세로, feature(흑백=1, 컬러=3))
# 모델 : None, 10, 10, 7
# model.add(Conv2D(7, (2,2), padding='same', input_shape=(28,28,1)))   # (None, 28, 28, 7)
model.add(Conv2D(7, (2,2), input_shape=(28,28,1)))   # (None, 27, 27, 7)
# model.add(Conv2D(16,(2,2)))
# model.add(MaxPooling2D(3,3))
# model.add(Conv2D(8,(2,2)))
# 다차원을 1차원으로 변경
model.add(Flatten())  # (None,5488)
model.add(Dense(10))  # (None,10)
model.add(Dense(10))

model.summary()