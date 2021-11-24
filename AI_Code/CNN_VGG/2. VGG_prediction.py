from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.applications.vgg16 import decode_predictions
from tensorflow.python.keras.applications.vgg16 import preprocess_input

from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import img_to_array

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 解决 Keras Bug:
# Exception ignored in: <bound method BaseSession.__del__ of <tensorflow.python.client.session.Session object at 0x0000029C2D0345F8>>
# Traceback (most recent call last):
#   File "C:\Users\LiuJ\Anaconda3\envs\my_tf\lib\site-packages\tensorflow\python\client\session.py", line 738, in __del__
# TypeError: 'NoneType' object is not callable
# 在代码末尾添加: K.clear_session()
from tensorflow.python.keras import backend as K

def vgg_predict():
    # 使用已训练好的 VGG 网络进行预测

    model = VGG16()
    model.summary() # 打印网络结果

    # 加载一个图片到VGG指定输入大小
    image = load_img('10_CNN/data/tiger.png', target_size=(224, 224))
    image.show() # 显示图片

    # 进行数据转换到numpy数组形式，以便于VGG能够进行使用
    image = img_to_array(image)

    # 形状修改
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # 输入数据进行预测,在此之前需要进行图片的归一化处理
    image = preprocess_input(image)
    y_predict = model.predict(image)

    # 进行结果解码
    label = decode_predictions(y_predict)
    print(label)
    # 进行lable获取
    res = label[0][0]
    # 预测的结果输出
    print('预测的类别为：%s 概率为：(%.2f%%)' % (res[1], res[2]*100))

    # 消除 Keras bug
    K.clear_session()

if __name__ == "__main__":

    vgg_predict()