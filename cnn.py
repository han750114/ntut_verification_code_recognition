import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, load_img

epochs = 10        # 训练的次数
img_rows = 38      # 验证码影像文件的高
img_cols = 135     # 验证码影像文件的宽
letters_in_img = 4  # 验证码影像文件中有几个字母
x_list = list()    # 存所有验证码数字影像文件的 array
y_list = list()    # 存所有的验证码数字影像文件 array 代表的正确数字
x_train = list()   # 存训练用验证码数字影像文件的 array
y_train = list()   # 存训练用验证码数字影像文件 array 代表的正确数字
x_test = list()    # 存测试用验证码数字影像文件的 array
y_test = list()    # 存测试用验证码数字影像文件 array 代表的正确数字

# 定义函数
def split_letters_in_img(img_array, x_list, y_list, img_filename):
    for i in range(letters_in_img):
        step = img_cols // letters_in_img
        x_list.append(img_array[:, i * step:(i + 1) * step] / 255)
        label = ord(img_filename[i]) - ord('A')
        label = max(0, min(label, 25))  # 确保标签在范围 [0, 25] 内
        y_list.append(label)

# 数据路径
data_path = '/Users/shuha/OneDrive/Desktop/tensorflow final project/datas'
img_filenames = os.listdir(data_path)

# 读取数据并进行处理
for img_filename in img_filenames:
    if '.png' not in img_filename:
        continue
    img = load_img(os.path.join(data_path, img_filename), color_mode='grayscale',
                   target_size=(img_rows, img_cols))
    img_array = img_to_array(img)
    split_letters_in_img(img_array, x_list, y_list, img_filename)

# 将标签转换为 one-hot 编码
y_list = keras.utils.to_categorical(y_list, num_classes=26)
# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x_list, y_list, test_size=0.2, random_state=42)

# 创建模型
model = models.Sequential()
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                        input_shape=(img_rows, img_cols // letters_in_img, 1)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(rate=0.25))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(26, activation='softmax'))

# 编译模型
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# 打印数据信息
print("x_train shape:", np.array(x_train).shape)
print("y_train shape:", np.array(y_train).shape)
print("x_train length:", len(x_train))
print("y_train length:", len(y_train))

# 训练模型
model.fit(np.array(x_train), np.array(y_train), batch_size=letters_in_img, epochs=epochs, verbose=1,
          validation_data=(np.array(x_test), np.array(y_test)))

# 保存或加载模型
if os.path.isfile('cnn_model.h5'):
    model = models.load_model('cnn_model.h5')
    print('Model loaded from file.')
else:
    model.save('cnn_model.h5')
    print('Model saved.')

