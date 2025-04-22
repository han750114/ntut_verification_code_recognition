import numpy as np
import cv2
import random
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

CHAR_SET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V',
            'W', 'X', 'Y', 'Z']
CHAR_SET_LEN = len(CHAR_SET)
CAPTCHA_LEN = 4

def text2label(text):
    label = np.zeros(CAPTCHA_LEN * CHAR_SET_LEN)
    for i in range(len(text)):
        idx = i * CHAR_SET_LEN + CHAR_SET.index(text[i])
        label[idx] = 1
    return label

# 獲取驗證碼圖片路徑及文字內容
def get_image_file_name(img_path):
    img_files = []
    img_labels = []
    for root, dirs, files in os.walk(img_path):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                img_files.append(os.path.join(root, file))
                img_labels.append(text2label(os.path.splitext(file)[0]))
    return img_files, img_labels

# 批量獲取資料
def get_next_batch(img_files, img_labels, batch_size):
    batch_x = np.zeros([batch_size, IMAGE_WIDTH * IMAGE_HEIGHT])
    batch_y = np.zeros([batch_size, CAPTCHA_LEN * CHAR_SET_LEN])

    for i in range(batch_size):
        idx = random.randint(0, len(img_files) - 1)
        file_path = img_files[idx]
        image = cv2.imread(file_path)
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0 / 255.0)
        batch_x[i, :] = image.flatten()
        batch_y[i, :] = img_labels[idx]

    return batch_x, batch_y

# 影象尺寸
IMAGE_HEIGHT = 38
IMAGE_WIDTH = 135

# 網路相關變數
X = tf.keras.Input(shape=(IMAGE_HEIGHT * IMAGE_WIDTH,), dtype=tf.float32)
Y = tf.keras.Input(shape=(CAPTCHA_LEN * CHAR_SET_LEN,), dtype=tf.float32)
keep_prob = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='keep_prob')  # dropout

def crack_captcha_cnn_network(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

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
    dense1 = tf.keras.layers.Dropout(rate=1 - keep_prob)(dense1)

    out = tf.keras.layers.Dense(CAPTCHA_LEN * CHAR_SET_LEN, activation='linear')(dense1)

    return out

# 驗證碼 CNN 網路

# 模型的相關引數
step_cnt = 200000  # 迭代輪數
batch_size = 16  # 批量獲取樣本數量
learning_rate = 0.0001  # 學習率

# 讀取驗證碼圖片集
img_path = '/Users/shuha/OneDrive/Desktop/tensorflow final project/datas'
img_files, img_labels = get_image_file_name(img_path)

# 劃分出訓練集、測試集
x_train, x_test, y_train, y_test = train_test_split(img_files, img_labels, test_size=0.2, random_state=33)

# 載入網路結構
output = crack_captcha_cnn_network()

# 損失函式、優化器
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

# 評估準確率
predict = tf.reshape(output, [-1, CAPTCHA_LEN, CHAR_SET_LEN])
max_idx_p = tf.argmax(predict, 2)
max_idx_l = tf.argmax(tf.reshape(Y, [-1, CAPTCHA_LEN, CHAR_SET_LEN]), 2)
correct_pred = tf.equal(max_idx_p, max_idx_l)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 訓練迴圈
for step in range(step_cnt):
    # 訓練模型
    batch_x, batch_y = get_next_batch(x_train, y_train, batch_size)
    with tf.GradientTape() as tape:
        logits = output(batch_x, training=True)
        loss_value = loss(batch_y, logits)
    grads = tape.gradient(loss_value, output.trainable_variables)
    optimizer.apply_gradients(zip(grads, output.trainable_variables))

    print('step:', step, 'loss:', loss_value.numpy())

    # 每100步評估一次準確率
    if step % 100 == 0:
        batch_x_test, batch_y_test = get_next_batch(x_test, y_test, batch_size)
        acc = accuracy(batch_x_test, batch_y_test)
        print('step:', step, 'acc:', acc.numpy())

        # 儲存模型
        output.save_weights('/Users/shuha/OneDrive/Desktop/tensorflow final project/crack_captcha', save_format='tf', overwrite=True)