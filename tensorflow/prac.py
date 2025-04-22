import numpy as np
import keras
from keras import applications
from keras.models import Sequential

from keras.layers.core import Dense
from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

train_path = "/Users/shuha/Downloads/kagglecatsanddogs_5340"
valid_path = "/Users/shuha/Downloads/kagglecatsanddogs_5340"
test_path = "/Users/shuha/Downloads/kagglecatsanddogs_5340"

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224, 224), classes=['dogs', 'cats'],
                                                         batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224, 224), classes=['dogs', 'cats'],
                                                         batch_size=4)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224, 224), classes=['dogs', 'cats'],
                                                        batch_size=10)

vgg16_model = keras.applications.vgg16.VGG16()

model = Sequential()
model.build(input_shape=(224,))
model.summary()

for layer in vgg16_model.layers:
    model.add(layer)
model.summary()


model.layers.pop()

for layer in model.layers:
    layer.trainable = False


model.add(Dense(2, activation='softmax'))
model.compile(Adam(lr=.00002122), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches, steps_per_epoch=10, validation_data=valid_batches, validation_steps=4, epochs=10,
                    verbose=2)

model.save("cat-dog-model-base-VGG16.h5")
