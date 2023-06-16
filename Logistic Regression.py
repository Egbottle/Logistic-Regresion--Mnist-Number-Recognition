import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 使用tensorflow加载mnist数据集
mnist = tf.keras.datasets.mnist

# x_train, y_train为训练集数据与对应标签; x_test,  y_test为测试集数据与对应标签
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 归一化处理：将像素点由0~255变为0~1
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0


# 将y进行独热编码
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

#使用keras建立10个逻辑回归模型，每个进行二分类，即指定数字为1，其他为0
models = []
for i in range(10):
    model = Sequential()
    model.add(Dense(1,activation="sigmoid"))
    models.append(model)

# 编译模型，采用随机梯度下降SGD
for model in models:
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

# 训练模型，将整个数据集训练10次，采用小批量随机梯度下降法
for i in range(10):
    model = models[i]
    model.fit(x_train, y_train[:, i], epochs=10, batch_size=100)
    print('模型',i,'训练完成')

# 在测试集上评估模型
for i in range(10):
   model = models[i]
   _, accuracy = model.evaluate(x_test, y_test[:, i])
   print('模型', i, '的准确率：', accuracy)


# 测试随机单张图片，输入图片序号
def test(n):
    d = x_test[n].copy()
    predictions = []
    for pre in models:
        predicton = pre.predict(np.expand_dims(x_test[n], axis=0))
        predictions.append(predicton)

    for i in range(10):
        if predictions[i] >= 0.5:
            print('预测的数字为',i)
            break
        elif i==9 and predictions[i] < 0.5:
            print("无法预测")
    print('真实数字为',np.argmax(y_test[n]))
    return i,np.argmax(y_test[n]),d

c = np.zeros(8).reshape(4,2)
c[0][0],c[0][1],t=test(8777)
c[1][0],c[1][1],y=test(76)
c[2][0],c[2][1],u=test(9548)
c[3][0],c[3][1],o=test(1014)


font = {'fontname': 'SimSun'}

fig, axes = plt.subplots(2, 2)

# 在每个子图中显示图片
axes[0, 0].imshow(t.reshape(28,28).astype('float64')*255.0)
axes[0, 0].set_title("预测的数字为：{}, 真实数字为：{}".format(c[0][0], c[0][1]), **font)

axes[0, 1].imshow(y.reshape(28,28).astype('float64')*255.0)
axes[0, 1].set_title("预测的数字为：{}, 真实数字为：{}".format(c[1][0], c[1][1]), **font)

axes[1, 0].imshow(u.reshape(28,28).astype('float64')*255.0)
axes[1, 0].set_title("预测的数字为：{}, 真实数字为：{}".format(c[2][0], c[2][1]), **font)

axes[1, 1].imshow(o.reshape(28,28).astype('float64')*255.0)
axes[1, 1].set_title("预测的数字为：{}, 真实数字为：{}".format(c[3][0], c[3][1]), **font)

# 调整子图之间的间距
plt.tight_layout()
plt.show()
