import numpy as np
import keras.optimizers
import keras.backend as K
from keras.datasets import mnist
# 資料庫
from keras.models import Sequential
from keras.layers import Dense, Dropout
# 隱藏層
from keras.utils import to_categorical
from keras.models import load_model

def Tanh(x):
	return (K.exp(x)-K.exp(-x))/(K.exp(x)+K.exp(-x)  )

def Trelu(x):
	return K.maximum(x,0)

def MAE(yHat, y):
	return 0.5*K.sum((y-yHat)**2)

def MSE(yHat, y):
	return 0.5*K.sum(abs(y-yHat))
# 載入MNIST資料集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data( )

# 建立人工神經網路
network = Sequential( )
network.add( Dense( 512, activation = Tanh, input_shape = ( 784, ) ) )
network.add(Dropout(0.2))
# 加隱藏層跟激活函數
network.add( Dense( 10, activation = 'softmax' ) )
opt = keras.optimizers.RMSprop(learning_rate=0.01)
network.compile( optimizer = opt, loss = MSE, metrics = ['accuracy'] )
print( network.summary() )
 
# 資料前處理
train_images = train_images.reshape( ( 60000, 28 * 28 ) )
# 六萬筆資料跟轉換成矩陣
train_images = train_images.astype( 'float32' ) / 255
# 正規化使每個數值在零到一之間(跑激活函數)
test_images = test_images.reshape( ( 10000, 28 * 28 ) )
test_images = test_images.astype( 'float32' ) / 255
train_labels = to_categorical( train_labels )
test_labels = to_categorical( test_labels )

# 訓練階段跟設置訓練次數
network.fit( train_images, train_labels, epochs = 5, batch_size = 200 )

# 測試階段
test_loss, test_acc = network.evaluate( test_images, test_labels )
print( "Test Accuracy:", test_acc )