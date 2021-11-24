import tensorflow as tf # 1.12.0
from tensorflow import keras # 2.1.6-tf

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class CNNMnist(object):
    def __init__(self):

        (self.x_train, self.y_train), (self.x_test, self.y_test) = \
            keras.datasets.cifar100.load_data()

        self.x_train = self.x_train / 255.0 # (50000, 32, 32, 3)
        self.x_test = self.x_test / 255.0 # (10000, 32, 32, 3)

        self.model = keras.Sequential([
            # 设置模型的输入层,不设置这一层的话，模型就可以处理任意大小的图片，网络宽度与输入大小相关，是动态变化的
            keras.layers.InputLayer(input_shape=(32, 32, 3)),

            keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same',
                                data_format='channels_last', activation=tf.nn.relu),
            keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same', data_format='channels_last'),
            keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same',
                                data_format='channels_last', activation=tf.nn.relu),
            keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same', data_format='channels_last'),
            keras.layers.Flatten(),
            keras.layers.Dense(units=1024, activation=tf.nn.relu),
            # keras.layers.Dropout(0.2),
            keras.layers.Dense(units=100, activation=tf.nn.softmax),
            ])

        self.model_path = "10_CNN/model/"

    def compile(self):
        self.model.compile(optimizer=keras.optimizers.Adam(), loss=tf.keras.losses.sparse_categorical_crossentropy,
                           metrics=['accuracy'])
        return None

    def fit(self):
        # 模型训练方式1：直接使用model.fit
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)

        # # 模型训练方式2：使用model.fit_generator，传入数据迭代器进行训练
        # # 使用数据迭代器可以对训练数据进行变换和增强
        # datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rotation_range=20,
        #                              width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, rescale=None)
        # datagen.fit(self.x_train)
        # data_loader = datagen.flow(self.x_train, self.y_train, shuffle=True, batch_size=32)
        # self.model.fit_generator(data_loader, epochs=1, validation_data=(self.x_test, self.y_test))
        #
        # # 模型训练方式3：使用model.fit方法时也可以传入数据迭代器
        # # 事实上，由源码可知，此时model.fit就是调用的model.fit_generator方法。
        # # 注意，此时model.fit方法中的batch_size不再起作用，训练的batch_size由datagen.flow方法中的设置确定。
        # self.model.fit(x=data_loader, epochs=1, validation_data=(self.x_test, self.y_test))


        # 模型保存
        self.model.save_weights(filepath=self.model_path + "Cifar_CNN.h5")

        return None

    def evaluate(self):

        if os.path.exists(self.model_path+"Cifar_CNN.h5"):
            self.model.load_weights(self.model_path+"Cifar_CNN.h5")
            print("模型已加载")

        test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test)

        print(test_loss, test_acc)

        return None

if __name__ == "__main__":
    my_model = CNNMnist()

    my_model.compile()

    my_model.fit()

    my_model.evaluate()