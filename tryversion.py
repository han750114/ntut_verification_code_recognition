import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# 定义必要的变量
img_rows = 38
img_cols = 135
letters_in_img = 4

# 加载模型
model = load_model('cnn_model.h5')

# 读取新的验证码图片
def preprocess_image(image_path, img_rows, img_cols, letters_in_img):
    img = load_img(image_path, color_mode='grayscale', target_size=(img_rows, img_cols))
    img_array = img_to_array(img)

    # 将图像切分为单个字母
    x_list = []
    for i in range(letters_in_img):
        step = img_cols // letters_in_img
        x_list.append(img_array[:, i * step:(i + 1) * step] / 255)

    return np.array(x_list)

# 替换 'your_image_path.png' 为你要识别的验证码图片的路径
image_path = "C:/Users/shuha/OneDrive/Desktop/tensorflow final project/datas/AAFW.png"

# 使用模型进行预测
predictions = model.predict(preprocess_image(image_path, img_rows, img_cols, letters_in_img))

# 获取最可能的结果
predicted_labels = np.argmax(predictions, axis=1)

# 将预测的数字标签转换为字母
predicted_letters = [chr(label + ord('A')) for label in predicted_labels]

# 打印结果
predicted_result = ''.join(predicted_letters)
print("Predicted result:", predicted_result)


