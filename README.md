# 🎓 NTUT 驗證碼識別 (CAPTCHA Recognition)

這是一個使用**卷積神經網路 (Convolutional Neural Network, CNN)** 來識別**四位數大寫英文字母**驗證碼的專案。

專案中提供了兩種不同的 CNN 模型實作，分別針對「單字元識別」和「整圖識別」策略進行訓練。

## 📁 檔案結構

```markdown

.
├── cnn.py                      \# 🧠 Keras Sequential API 模型訓練腳本 (單字元識別)
├── training.py                 \# 🧠 原生 TensorFlow 函式 API 模型訓練腳本 (整圖識別)
├── cnn\_model.h5                \# cnn.py 訓練後儲存的模型檔案 (如果存在)
├── 1600張驗證碼訓練資料雲端連結.txt \# 訓練資料集下載連結
└── datas/                      \# 驗證碼圖片存放目錄 (例如: AAFW.png, ACFV.png 等)
└── ...

````

---

## 💡 模型概述

本專案提供了兩種不同的 CNN 實作方法來解決驗證碼識別問題：

| 檔案名稱 | 框架 / API | 識別策略 | 影像輸入尺寸 (單個樣本) | 輸出類別數 |
| :--- | :--- | :--- | :--- | :--- |
| `cnn.py` | TensorFlow + Keras Sequential | **單字元識別** (將圖片切割成 4 個字元後分別訓練) | $38 \times 33$ 像素 (單字元) | 26 (字母 A-Z) |
| `training.py` | 原生 TensorFlow 函式 API + OpenCV (cv2) | **整圖識別** (直接預測全部 4 個字元) | $38 \times 135$ 像素 (扁平化為 $5130$ 維向量) | $4 \times 26 = 104$ |

### 1. `cnn.py` (Keras - 單字元模型)

此模型專門針對單個字元進行分類，每次輸入一個切割後的字元圖片。

#### 模型結構 (簡化版)

* `Conv2D` (32 filters, $3\times3$)
* `Conv2D` (64 filters, $3\times3$)
* `MaxPooling2D` ($2\times2$)
* `Dropout` (rate=0.25)
* `Flatten`
* `Dense` (128 units, `relu`)
* `Dropout` (rate=0.5)
* `Dense` (26 units, `softmax` activation)

#### 訓練細節

* **訓練次數 (epochs):** 10
* **批次大小 (batch\_size):** 4
* **優化器:** Adam
* **損失函數:** Categorical Crossentropy

### 2. `training.py` (TensorFlow - 整圖模型)

此模型讀取完整的驗證碼圖片，並輸出一個包含所有四個字元預測結果的向量。

#### 模型結構 (`crack_captcha_cnn_network`)

* **輸入層:** 扁平化的 $38 \times 135$ 像素影像
* 3 組 (`Conv2D`, `MaxPooling2D`, `BatchNormalization`) 層
* `Flatten`
* `Dense` (1024 units, `relu` activation)
* `Dropout` (使用 `keep_prob` 輸入控制)
* **輸出層:** `Dense` ($4 \times 26 = 104$ units, `linear` activation)

#### 訓練細節

* **迭代輪數 (step\_cnt):** 200,000
* **批次大小 (batch\_size):** 16
* **學習率 (learning\_rate):** 0.0001
* **損失函數:** Sigmoid Cross-Entropy with Logits

---

## 🛠️ 運行環境與依賴

### 軟體要求

* Python (建議 3.x)
* TensorFlow / Keras (版本應與程式碼相容)
* Numpy
* Scikit-learn
* OpenCV (`cv2`)

### 安裝依賴項

您可以使用 `pip` 安裝主要的依賴項：

```bash
pip install tensorflow numpy scikit-learn opencv-python
````

-----

## 💾 資料集

| 項目 | 說明 |
| :--- | :--- |
| **圖片數量** | 1600 張驗證碼圖片 |
| **內容** | 每張圖片由 **4 個大寫英文字母**組成 |
| **圖片尺寸** | $38 \times 135$ 像素 |
| **字元集** | 26 個大寫英文字母 (A-Z) |
| **下載連結** | 請參閱 `1600張驗證碼訓練資料雲端連結.txt` |
| **資料存放路徑** | 圖片需放置在 **`datas/`** 目錄下，路徑需與程式碼中指定的一致 (例如：`/Users/shuha/OneDrive/Desktop/tensorflow final project/datas`)。 |

-----

## 🚀 使用指南

### 訓練模型

請選擇一個訓練腳本並運行：

#### 1\. 使用 Keras 模型訓練 (`cnn.py`)

此腳本將圖片切分成單個字元進行訓練。模型訓練完成後，將會儲存或載入 `cnn_model.h5` 檔案。

```bash
python cnn.py
```

#### 2\. 使用原生 TensorFlow 模型訓練 (`training.py`)

此腳本將訓練一個處理完整圖片的 CNN 模型。模型權重將會定期儲存到指定路徑（例如：`/Users/shuha/OneDrive/Desktop/tensorflow final project/crack_captcha`）。

```bash
python training.py
```

```
```
