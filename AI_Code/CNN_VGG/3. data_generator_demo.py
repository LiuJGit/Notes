from tensorflow.python.keras.datasets import cifar100
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

def data_gen_demo():
    # 数据集
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    # Generate batches of tensor image data with real-time data augmentation.
    # The data will be looped over (in batches).
    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rotation_range=20,
                                  width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

    # Fits the data generator to some sample data.
    # Only required if `featurewise_center` or `featurewise_std_normalization` or `zca_whitening` are set to True.
    # When `rescale` is set to a value, rescaling is applied to sample data before computing the internal data stats.
    # 也就是提供batch数据前，需要先提供the data-dependent transformations所依赖的数据
    datagen.fit(x_train)

    # Takes data & label arrays, generates batches of augmented data.
    # 还可以使用 save_to_dir 等参数保存生成的文件到本地
    # datagen.flow()类似于pytorch的DataLoader，返回一个数据迭代器
    # 当数据是从本地读取时，对应的方法为datagen.flow_from_directory()
    for x_batch, y_batch in datagen.flow(x_train, y_train,shuffle=True,batch_size=64):
        # we need to break the loop by hand because
        # the generator loops indefinitely
        # 在pytorch中，dataloader将dataset的所有数据遍历完一遍后便不再提供数据，循环停止，
        # 但这里，循环不会自动停止，而是无限制地进行下去。这也不难理解，因为存在数据增强，
        # 程序总能实时生成新的增强样本，因此循环可以无限进行下去
        print(x_batch.shape,y_batch.shape)
        break

    # for e in range(epochs):
    #     print('Epoch', e)
    #     batches = 0
    #     for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
    #         model.fit(x_batch, y_batch)
    #         batches += 1
    #         if batches >= len(x_train) / 32:
    #             # we need to break the loop by hand because
    #             # the generator loops indefinitely
    #             break

    # Keras还提供了fit_generator方法供model调用，由此可避免训练时用户基于fit方法手动实现上述复杂循环。
    # model.fit_generator()直接传入数据迭代器，用于评估的validation_data也可以接受数据元组，具体使用细节可参考代码注释。
    # model.fit_generator(generator=train_gen, epochs=1, validation_data=test_gen, callbacks=[check])

    # 事实上，model.fit方法除了可以接受由数据迭代器提供的数据，也可以直接接受数据迭代器datagen.flow()/flow_from_directory()。
    # 接受迭代器时，target也从x中获取，参数y不传值。
    # 事实上，model.fit的源码此时就是调用的model.fit_generator方法。
    # 此时model.fit方法中的batch_size不再起作用，训练的batch_size由datagen.flow/flow_from_directory()方法中的设置确定。

    return None

def data_gen_demo2():

    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rotation_range=20,
                                 width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

    # bug:`featurewise_center`以及`featurewise_std_normalization`均为True，但手头没有数据供datagen进行fit
    data_gen = datagen.flow_from_directory(directory="10_CNN/data/train/",
                                           target_size=(224, 224),
                                           batch_size=32,
                                           class_mode='binary',
                                           shuffle=True)

    for x_batch,y_batch in data_gen:
        print(x_batch.shape,y_batch.shape)
        break



if __name__ == "__main__":
    data_gen_demo()
    # data_gen_demo2()

