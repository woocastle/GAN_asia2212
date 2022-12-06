import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from keras.datasets import mnist

input_img = Input(shape=(28, 28, 1)) # padding을 써줬기에 28이 그대로 출력 # 줄이기 # 마지막꺼 1컬러
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPool2D((2, 2), padding='same')(x) # 사이즈를 줄일 때 # 4개의 값중 가장 작은 픽셀로 줄임
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPool2D((2, 2), padding='same')(x) # maxpool은 건너 뛰고 씌워진다. padding을 쓰면 마지막에 0의 값을 채워줌
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPool2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x) # 사이즈를 키울 때 # 한칸을 2x2로 만든다. 그래서 4칸짜리가 8칸이 됌
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x) # 사이즈를 키울 때
x = Conv2D(16, (3, 3), activation='relu')(x) # 여기서 14가 된다 결국 사이즈를 맞춰줌
x = UpSampling2D((2, 2))(x) # 사이즈를 키울 때
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x) # 마지막 출력을 1장만 준다. # 마지막에 값이 0,1이기에

autoencoder = Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data() # 비지도 학습이라 따로 필요가 없음
x_train = x_train / 255 # 나누고 그 값을 다시 저장하는 것, 복합연산자
x_test = x_test / 255 # 위나 밑이나 똑같은거임
# print(x_train.shape)
# print(x_train[0])
conv_x_train = x_train.reshape(-1, 28, 28, 1) # 3개의 컬러 묶음 28*28로 묶어넣은 것
conv_x_test = x_test.reshape(-1, 28, 28, 1) # reshape 묵는법 (12, ) -> (6, 2) / (-1, 3)을 주면 알아서 묶어줌
# print(conv_x_train.shape)
# print(conv_x_train)


noise_factor = 0.5 # 잡음의 크기를 조절하기 위해서 사용
conv_x_train_noisy = conv_x_train + np.random.normal(0, 1, size=conv_x_train.shape) * noise_factor
#잡음은 정규분포를 따른다. 똑같은 모양으로 shape을 준다.
conv_x_train_noisy = np.clip(conv_x_train_noisy, 0.0, 1.0) # 0.0넘어가면 0.0, 1.0넘어가면 1.0 상한 하한값 줌
conv_x_test_noisy = conv_x_test + np.random.normal(0, 1, size=conv_x_test.shape) * noise_factor
#잡음은 정규분포를 따른다. 똑같은 모양으로 shape을 준다.
conv_x_test_noisy = np.clip(conv_x_test_noisy, 0.0, 1.0) # 0.0넘어가면 0.0, 1.0넘어가면 1.0 상한 하한값 줌
plt.figure(figsize=(20, 4))
n = 10
for i in range(n):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 1 + n)
    plt.imshow(conv_x_test_noisy[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

fit_hist = autoencoder.fit(conv_x_train_noisy, conv_x_train, epochs=50,
                batch_size=256, validation_data=(conv_x_test_noisy, conv_x_test))

autoencoder.save('./models/autoencoder_noisy.h5')

decoded_img = autoencoder.predict(x_test[:10])


plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, 10, i + 1) # 첫번째줄
    plt.imshow(conv_x_test_noisy[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 1 + n) # 두번째줄
    plt.imshow(decoded_img[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.show()



