import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from keras.datasets import mnist

input_img = Input(shape=(784,)) # 밑에 32는 픽셀을 의미함
encoded = Dense(32, activation='relu') # input을 따로 안줌 위에 줌 # relu는 넘어간다.
encoded = encoded(input_img)
decoded = Dense(784, activation='sigmoid') # 위에 encoded를 뱉어낸다. # minmax정규화를 함
decoded = decoded(encoded)
autoencoder = Model(input_img, decoded)
autoencoder.summary()

encoder = Model(input_img, encoded)
encoder.summary()

encoder_input = Input(shape=(32,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoder_input, decoder_layer(encoder_input))
decoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data() # 비지도 학습이라 따로 필요가 없음
x_train = x_train / 255 # 나누고 그 값을 다시 저장하는 것, 복합연산자
x_test = x_test / 255 # 위나 밑이나 똑같은거임

flatted_x_train = x_train.reshape(-1, 784)
flatted_x_test = x_test.reshape(-1, 784)

fit_hist = autoencoder.fit(flatted_x_train, flatted_x_train, epochs=50,
                batch_size=256, validation_data=(flatted_x_test, flatted_x_test))

encoded_img = encoder.predict(x_test[:10].reshape(-1, 784))
decoded_img = decoder.predict(encoded_img)

n = 10
#plt.gray()
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, 10, i + 1) # 첫번째줄
    plt.imshow(x_test[i])
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



