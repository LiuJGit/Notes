from keras.layers import Dense, Input, Conv2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from tensorflow.python.keras.applications import VGG16

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 解决 Keras Bug:
from tensorflow.python.keras import backend as K

def test_GAP2D():
    x = Input(shape=[8, 8, 2048])
    print(x) # Tensor("input_1:0", shape=(?, 8, 8, 2048), dtype=float32)
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    print(x) # Tensor("avg_pool/Mean:0", shape=(?, 2048), dtype=float32)
    x = Dense(1000, activation='softmax', name='predictions')(x)  # shape=(?, 1000)
    print(x) # Tensor("predictions/Softmax:0", shape=(?, 1000), dtype=float32)
    return None

def test_VGG():
    VGG_model = VGG16()
    base_model = VGG16(weights='imagenet', include_top=False)
    # 可以看到，二者的网络大小是不同的，VGG_model中各层输出的shape，除了batch_size，其余大小都是确定的，
    # 而base_model只是采用了与之相同的核，也就是说其各层输出的shape只有最后的channel的大小是确定的，
    # 与VGG_model相同，而其他阶的维数都是待定的
    VGG_model.summary()
    base_model.summary()

    print(VGG_model.inputs) # [<tf.Tensor 'input_1:0' shape=(?, 224, 224, 3) dtype=float32>]
    print(base_model.inputs) # [<tf.Tensor 'input_2:0' shape=(?, ?, ?, 3) dtype=float32>]
    print(base_model.outputs[0]) # Tensor("block5_pool_1/MaxPool:0", shape=(?, ?, ?, 512), dtype=float32)



    # 消除 Keras bug
    K.clear_session()

if __name__ == "__main__":
    test_VGG()