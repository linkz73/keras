# CNN 은 특징(feature) 를 특징있게 잡아내는 것
from keras.models import Sequential

filter_size = 32
kernel_size = (3,3)

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
model = Sequential()
'''
conv2D : 이미지, conv3D : 3차원, conv1D : 자주 사용(시계열, 증권 등)
filter : 컨벌루션 필터 수(아웃풋 갯수), 
kernel_size : 자르는 단위, 
strides : 몇 칸씩 이동 (디폴트는 1) 
padding : 경계 처리 방법 (same-입력이미지 사이즈와 동일, valid-유효영역만 출력), 디폴트는 valid
   - same 일 경우 사이드에 0을 씌워서 shape를 인풋과 동일하게 맞춰 줌
     same 을 사용하는 이유 : 양쪽 끝이 훈련시 가중치가 떨어지는 것을 개선하기 위해, 가장자리 손실치 손해를 덜 볼수 있도록 하기 위해.
input shape : 픽셀 가로, 픽셀 세로, feature(흑백=1, 컬러=3)
'''
# model.add(Conv2D(7, (2,2), padding='same', input_shape=(28,28,1)))   # (None, 28, 28, 7)
# model.add(Conv2D(7, (2,2), input_shape=(28,28,1)))   # (None, 27, 27, 7)
# model.add(Conv2D(7, (3,3), input_shape=(5,5,1)))   # (None, 3, 3, 7)
# model.add(Conv2D(3, (2,2), input_shape=(5,5,1)))   # (None, 4, 4, 3)
# model.add(Conv2D(4, (2,2)))   # (None, 3, 3, 4)

model.add(Conv2D(3, (3,3), padding='same', input_shape=(28,28,1)))   # (None, 28, 28, 3)
model.add(Conv2D(4, (2,2)))  # (None, 27, 27, 4)
model.add(Conv2D(16,(2,2)))  # (None, 26, 26, 16)
# model.add(MaxPooling2D(3,3))
model.add(Conv2D(8,(2,2)))  # (None, 25, 25, 8)
# CNN 을 dense 로 전환하기 위해 shape차원을 맞춰야 함. 다차원을 1차원으로 변경
model.add(Flatten())  # (None,5000)
model.add(Dense(10))  # (None,10)
model.add(Dense(1))  # (None,1)

model.summary()