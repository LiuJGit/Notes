import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras import Model
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.applications.vgg16 import preprocess_input
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TransferModel(object):

    def __init__(self):

        self.image_size = (224, 224)
        self.train_dir = "10_CNN/data/train/"
        self.test_dir = "10_CNN/data/test/"
        self.batch_size = 32

        # Generate batches of tensor image data with real-time data augmentation.
        #    The data will be looped over (in batches).
        # 参数设置及举例可直接参考代码注释
        self.train_generator = ImageDataGenerator(rescale=1.0 / 255)
        self.test_generator = ImageDataGenerator(rescale=1.0 / 255)

        # 定义迁移学习的基类模型
        # weights='imagenet' 表示 pre-training on ImageNet
        # include_top: whether to include the 3 fully-connected layers at the top of the network.
        self.base_model = VGG16(weights='imagenet', include_top=False)

        # 图片类别字典:{num:class_name}
        self.label_dict = {}

    def get_local_data(self):
        """
        读取本地的图片数据以及类别
        :return:训练数据和测试数据迭代器
        """
        # 使用flow_from_derectory
        train_gen = self.train_generator.flow_from_directory(directory=self.train_dir,
                                                             target_size=self.image_size,
                                                             batch_size=self.batch_size,
                                                             class_mode='binary',
                                                             shuffle=True)
        test_gen = self.test_generator.flow_from_directory(directory=self.test_dir,
                                                           target_size=self.image_size,
                                                           batch_size=self.batch_size,
                                                           class_mode='binary',
                                                           shuffle=True)

        # 构建类别字典，将train_gen.class_indices字典的键值对互换即可
        # self.label_dict = {value:key for key,value in train_gen.class_indices.items()}
        # 另一种实现方式
        self.label_dict = dict(zip(train_gen.class_indices.values(),train_gen.class_indices.keys()))

        print(self.label_dict)

        return train_gen, test_gen

    def refine_base_model(self):
        """
        微调VGG结构，5blocks后面+全局平均池化（减少迁移学习的参数数量）+两个全连接层
        :return:
        """

        # 1、获取原notop模型得出
        # [<tf.Tensor 'block5_pool/MaxPool:0' shape=(?, ?, ?, 512) dtype=float32>]
        x = self.base_model.outputs[0] # self.base_model.outputs是一个列表，为了得到tensor，所以使用索引[0]

        # 2、在输出后面增加全局池化等自定义结构
        x = keras.layers.GlobalAveragePooling2D()(x) # [?, ?, ?, 512]---->[?, 1 * 1 * 512]
        x = keras.layers.Dense(1024, activation=tf.nn.relu)(x)
        y_predict = keras.layers.Dense(5, activation=tf.nn.softmax)(x)

        # 定义新模型
        # 输入：VGG 模型的输入， 输出：y_predict
        self.transfer_model = Model(inputs=self.base_model.inputs, outputs=y_predict)

        return None

    def freeze_model(self):
        """
        冻结VGG模型（5blocks）
        冻结VGG的多少，可根据数据量确定
        :return:
        """
        for layer in self.base_model.layers:
            layer.trainable = False

        return None

    def model_compile(self):
        self.transfer_model.compile(optimizer=keras.optimizers.Adam(),
                                    loss=keras.losses.sparse_categorical_crossentropy,
                                    metrics=['accuracy'])
        return None

    def fit_gen(self, train_gen, test_gen):
        """
        训练模型，model.fit_generator()不是选择model.fit()
        """
        # 每一次迭代记录准确率的h5文件
        check = keras.callbacks.ModelCheckpoint('10_CNN/model/transfer_{epoch:02d}-{val_acc:.2f}.h5',
                                                monitor='val_loss',  # 需要监视的值
                                                save_best_only=True,
                                                save_weights_only=True,
                                                mode='auto',
                                                period=1)

        self.transfer_model.fit_generator(train_gen, epochs=1, validation_data=test_gen, callbacks=[check])

        # model.save_weights("./Transfer.h5")

    def model_predict(self):
        """
        预测类别
        :return:
        """

        # 设置类别字典
        self.label_dict = {0: 'bus', 1: 'dinosaurs', 2: 'elephants', 3: 'flowers', 4: 'horse'}

        # 加载模型，transfer_model
        self.transfer_model.load_weights("10_CNN/model/transfer_01-0.88.h5")

        # 读取图片，处理
        image = load_img("10_CNN/data/400.jpg", target_size=(224, 224))
        image = img_to_array(image)
        # print(image.shape) # (224, 224, 3)
        # 四维(224, 224, 3)--->(1， 224， 224， 3)
        img = image.reshape([1, image.shape[0], image.shape[1], image.shape[2]])

        # 预处理
        # 讲义中使用的是ImageNet的预处理image = preprocess_input(img)，但个人认为只需除以255即可
        image = img / 255

        # 预测
        predictions = self.transfer_model.predict(image)
        # print('predictions',predictions)

        # 对预测结果进行处理，得到预测类别
        res = np.argmax(predictions, axis=1)
        # print(res)

        print('类别为：{}，概率为：{:.2f}%'.format(self.label_dict[res[0]], predictions[0][res[0]]*100))

        return None



if __name__ == '__main__':

    tm = TransferModel()
    # print(tm.base_model.summary())

    # # 获取数据迭代器
    # train_gen, test_gen = tm.get_local_data()
    # # print(train_gen)
    # # for data in train_gen:
    # #     print(data[0].shape, data[1].shape)
    # #     break
    # # 构建transfer_model
    # tm.refine_base_model()
    # # print(tm.transfer_model)
    # # 冻结base_model
    # tm.freeze_model()
    # # 模型编译
    # tm.model_compile()
    # # 模型训练
    # tm.fit_gen(train_gen, test_gen)

    # 预测
    # 构建transfer_model
    tm.refine_base_model()
    # 模型加载及预测
    tm.model_predict()