import numpy as np
import cv2
import random
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

CHAR_SET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V',
            'W', 'X', 'Y', 'Z']
CHAR_SET_LEN = len(CHAR_SET)
CAPTCHA_LEN = 4


def text2label(text):
    label = np.zeros(CAPTCHA_LEN * CHAR_SET_LEN)
    for i in range(len(text)):
        idx = i * CHAR_SET_LEN + CHAR_SET.index(text[i])
        label[idx] = 1
    return label


def get_image_file_name(img_path):
    img_files = []
    img_labels = []
    for root, dirs, files in os.walk(img_path):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                img_files.append(os.path.join(root, file))
                img_labels.append(text2label(os.path.splitext(file)[0]))
    return img_files, img_labels


def get_next_batch(img_files, img_labels, batch_size):
    batch_x = np.zeros([batch_size, IMAGE_WIDTH * IMAGE_HEIGHT * 3])
    batch_y = np.zeros([batch_size, CAPTCHA_LEN * CHAR_SET_LEN])

    for i in range(batch_size):
        idx = random.randint(0, len(img_files) - 1)
        file_path = img_files[idx]
        image = cv2.imread(file_path)
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0 / 255.0)
        flattened_image = image.flatten()

        batch_x[i, :] = flattened_image
        batch_y[i, :] = img_labels[idx]

    return batch_x, batch_y


IMAGE_HEIGHT = 38
IMAGE_WIDTH = 135

X = tf.keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.float32)
Y = tf.keras.Input(shape=(CAPTCHA_LEN * CHAR_SET_LEN,), dtype=tf.float32)
keep_prob_scalar = 0.5


def crack_captcha_cnn_network(w_alpha=0.01, b_alpha=0.1):
    x = tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.float32)

    conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    conv1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)

    conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv1)
    conv2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)

    conv3 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv2)
    conv3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)

    flatten = tf.keras.layers.Flatten()(conv3)

    dense1 = tf.keras.layers.Dense(1024, activation='relu')(flatten)
    dense1 = tf.keras.layers.Dropout(rate=1 - keep_prob_scalar)(dense1)

    out = tf.keras.layers.Dense(CAPTCHA_LEN * CHAR_SET_LEN, activation='linear')(dense1)

    model = tf.keras.models.Model(inputs=x, outputs=out)

    return model


step_cnt = 200000
batch_size = 16
learning_rate = 0.0001

img_path = '/Users/shuha/OneDrive/Desktop/tensorflow final project/datas'
img_files, img_labels = get_image_file_name(img_path)

x_train, x_test, y_train, y_test = train_test_split(img_files, img_labels, test_size=0.2, random_state=33)

# 載入網路結構
model = crack_captcha_cnn_network()

# 編譯模型
logits = model(X, training=True)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

# 評估準確率
predict = tf.reshape(logits, [-1, CAPTCHA_LEN, CHAR_SET_LEN])
max_idx_p = tf.argmax(predict, 2)
max_idx_l = tf.argmax(tf.reshape(Y, [-1, CAPTCHA_LEN, CHAR_SET_LEN]), 2)
correct_pred = tf.equal(max_idx_p, max_idx_l)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 編譯模型
model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])

# 保存最佳準確率
best_accuracy = 0.0

for step in range(step_cnt):
    batch_x, batch_y = get_next_batch(x_train, y_train, batch_size)

    with tf.GradientTape() as tape:
        logits = model(batch_x, training=True)
        loss_value = loss(batch_y, logits)

    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print('step:', step, 'loss:', loss_value.numpy())

    if step % 100 == 0:
        batch_x_test, batch_y_test = get_next_batch(x_test, y_test, batch_size)
        logits_test = model(batch_x_test, training=False)
        max_idx_p_test = tf.argmax(tf.reshape(logits_test, [-1, CAPTCHA_LEN, CHAR_SET_LEN]), 2)
        max_idx_l_test = tf.argmax(tf.reshape(batch_y_test, [-1, CAPTCHA_LEN, CHAR_SET_LEN]), 2)
        correct_pred_test = tf.equal(max_idx_p_test, max_idx_l_test)
        acc_test = tf.reduce_mean(tf.cast(correct_pred_test, tf.float32))
        print('step:', step, 'acc:', acc_test.numpy())

        if acc_test > best_accuracy:
            best_accuracy = acc_test.numpy()
            model.save_weights('/Users/shuha/OneDrive/Desktop/tensorflow final project/crack_captcha', save_format='tf',
                               overwrite=True)
