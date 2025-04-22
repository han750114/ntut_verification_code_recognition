from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
import numpy as np

# 載入MNIST資料集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 資料前處理
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 將訓練資料和標籤合併
data = np.hstack((train_images, train_labels))

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_index, val_index in kfold.split(data, train_labels.argmax(axis=1)):

    # 建立人工神經網路
    network = Sequential()
    network.add(Dense(512, activation='relu', input_shape=(784,)))
    network.add(Dense(10, activation='softmax'))
    network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print(network.summary())

    train_data, val_data = data[train_index], data[val_index]
    train_images_fold, train_labels_fold = train_data[:, :-10], train_data[:, -10:]
    val_images_fold, val_labels_fold = val_data[:, :-10], val_data[:, -10:]

    network.fit(train_images_fold, train_labels_fold, epochs=5, batch_size=200)

    val_loss, val_acc = network.evaluate(val_images_fold, val_labels_fold)
    print("Validation Accuracy:", val_acc)

# 測試階段
test_loss, test_acc = network.evaluate(test_images, test_labels)
print("Test Accuracy:", test_acc)