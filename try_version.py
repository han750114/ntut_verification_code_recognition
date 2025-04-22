import tkinter as tk
from PIL import ImageGrab, ImageTk
from tkinter import messagebox
import os
from datetime import datetime
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

class ScreenshotApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Screenshot & Recognition App")

        self.start_x = tk.IntVar(value=100)
        self.start_y = tk.IntVar(value=100)
        self.width = tk.IntVar(value=200)
        self.height = tk.IntVar(value=200)

        # 加載深度學習模型
        self.model = load_model('cnn_model.h5')

        self.create_widgets()

        # 綁定鍵盤事件，改為按空白鍵觸發
        self.master.bind("<space>", self.take_screenshot)

    def create_widgets(self):
        # 矩形範圍輸入
        tk.Label(self.master, text="起始X座標:").pack()
        tk.Entry(self.master, textvariable=self.start_x).pack()

        tk.Label(self.master, text="起始Y座標:").pack()
        tk.Entry(self.master, textvariable=self.start_y).pack()

        tk.Label(self.master, text="寬度:").pack()
        tk.Entry(self.master, textvariable=self.width).pack()

        tk.Label(self.master, text="高度:").pack()
        tk.Entry(self.master, textvariable=self.height).pack()

        # 提示信息
        tk.Label(self.master, text="按空白鍵截圖並進行辨識").pack(pady=10)

    def take_screenshot(self, event=None):
        # 獲取截圖
        screenshot = ImageGrab.grab(bbox=(self.start_x.get(), self.start_y.get(),
                                          self.start_x.get() + self.width.get(),
                                          self.start_y.get() + self.height.get()))

        # 使用當前的日期和時間來構建檔名
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"screenshot_{current_time}.png"

        # 指定圖片保存的路徑
        save_path = '/Users/shuha/OneDrive/Desktop/tensorflow final project/Screenshots'

        # 保存圖片，指定擴展名為 .png
        screenshot.save(os.path.join(save_path, file_name), format='PNG')

        # 進行深度學習模型的預測
        preprocessed_image = self.preprocess_image(os.path.join(save_path, file_name))
        predictions = self.model.predict(preprocessed_image)
        predicted_labels = np.argmax(predictions, axis=1)
        predicted_letters = [chr(label + ord('A')) for label in predicted_labels]
        predicted_result = ''.join(predicted_letters)

        # 顯示成功訊息
        messagebox.showinfo("成功", f"截圖已保存到桌面: {file_name}\n預測結果: {predicted_result}")

    def preprocess_image(self, image_path):
        img = load_img(image_path, color_mode='grayscale', target_size=(38, 135))
        img_array = img_to_array(img)

        # 將圖像切分為單個字母
        x_list = []
        for i in range(4):  # 4是驗證碼中的字母數
            step = 135 // 4
            x_list.append(img_array[:, i * step:(i + 1) * step] / 255)

        return np.array(x_list)

# 建立主視窗
root = tk.Tk()
app = ScreenshotApp(root)
root.mainloop()


