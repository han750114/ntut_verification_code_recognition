from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

#載入MNIST資料集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#建立人工神經網路
network = Sequential()
network.add(Dense(512, activation='relu', input_shape=(784,)))
network.add(Dropout(0.5))
network.add(Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                metrics=['accuracy'])
print(network.summary())

#資料前處理
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#僅使用前5000筆
train_images = train_images[:5000]
train_labels = train_labels[:5000]

#訓練階段
history = network.fit(train_images, train_labels, validation_data=(test_images, test_labels) , epochs= 5, batch_size=200)
test_loss, test_acc = network.evaluate( test_images, test_labels )
print( "Test Accuracy:", test_acc )

import matplotlib.pyplot as plt

loss = history.history["loss"]
val_loss = history.history["val_loss"]
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
epochs = range(1, len(loss)+1)
#plt.plot(epochs, loss, "b-o", label="Training Loss")
plt.plot(epochs, acc, "b-o", label="Training Accuracy")
#plt.plot(epochs, val_loss, "r--x", label="Validation Loss")
plt.plot(epochs, val_acc, "r--x", label="Validation Accuracy")
plt.title("Training and Validation")
plt.xlabel("Epochs")
plt.ylabel("Loss and Accuracy")
plt.legend()
plt.show()