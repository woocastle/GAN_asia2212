import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from keras.datasets import mnist
import os

OUT_DIR = './DNN_out'
img_shape = (28, 28, 1)
epochs = 100000
batch_size = 128
noise = 100
sample_interval = 100

(x_train, _), (_, _) = mnist.load_data()
print(x_train.shape)

x_train = x_train / 127.5 - 1
x_train = np.expand_dims(x_train, axis=3) # reshap (-1, 28, 28, 1)이랑 결과는 같다.
print(x_train.shape)

generator = Sequential()
generator.add(Dense(128, input_dim=noise))
generator.add(LeakyReLU(alpha=0.01)) # relu를 레이어에 추가했음 # 진하게 칠해진 부분에 대해서만 강하게 반응하기 위해 leakyrelu사용
generator.add(Dense(784, activation='tanh'))
generator.add(Reshape(img_shape))
generator.summary()

lrelu = LeakyReLU(alpha=0.01) # leakyrelu는 alpha 값을 무조건 줘야한다.

discriminator = Sequential()
discriminator.add(Flatten(input_shape=img_shape))
discriminator.add(Dense(128, activation=lrelu)) # 이렇게 넣어주려면 미리 만들어 줘야한다.
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.summary() # 얘는 2진분류기를 쓰면 된다.
discriminator.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

gan_model = Sequential()
gan_model.add(generator)
gan_model.add(discriminator)
gan_model.summary()
gan_model.compile(loss='binary_crossentropy', optimizer='adam')

real = np.ones((batch_size, 1))
print(real)
fake = np.zeros(((batch_size, 1)))
print(fake)

for epoch in range(epochs):
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_imgs = x_train[idx] # 실제 이미지

    z = np.random.normal(0, 1, (batch_size, noise))
    fake_imgs = generator.predict(z)

    d_hist_real = discriminator.train_on_batch(real_imgs, real)
    d_hist_fake = discriminator.train_on_batch(fake_imgs, fake)

    d_loss, d_acc = np.add(d_hist_fake, d_hist_real) * 0.5 # 두개를 평균 냈다.

    discriminator.trainable = False

    if epoch % 2 == 0:
        z = np.random.normal(0, 1, (batch_size, noise))
        gan_hist = gan_model.train_on_batch(z, real)

    if epoch % sample_interval == 0:
        print('%d, [D loss: %f, acc.: %.2f%%],[G loss: %f]'%(
            epoch, d_loss, d_loss, gan_hist))
        row = col = 4
        z = np.random.normal(0, 1, (row * col, noise))
        fake_imgs = generator.predict(z)
        fake_imgs = 0.5 * fake_imgs + 0.5
        _, axs = plt.subplots(row, col, figsize=(5, 5), sharey=True, sharex=True)
        cont = 0
        for i in range(row):
            for j in range(col):
                axs[i, j].imshow(fake_imgs[cont, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cont += 1
        path = os.path.join(OUT_DIR, 'img-{}'.format(epoch+1)) # 합치느냐 os를 씀
        plt.savefig(path)
        plt.close()

