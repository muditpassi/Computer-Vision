import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten

mnist = tf.keras.datasets.mnist

print("Load Data..")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(len(x_train), "Train Sequences")
print(len(x_test), "Test Sequences")
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train, x_test = x_train/255.0, x_test/255.0

print("Building Model..")
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=[5, 5], input_shape=(28, 28, 1), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=[2, 2], strides=2))
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=[2, 2], strides=2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Train Data..")
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, batch_size=32)
print("Score", score[0], "accuracy", score[1])
