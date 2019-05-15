# 导包
import os

from load_train_test_data import load_train_valid_data
from paths import root_dir, mkdir_if_not_exist

from keras_preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model

from keras.callbacks import TensorBoard, ReduceLROnPlateau,ModelCheckpoint

from keras import regularizers
from keras.optimizers import Adam

# 指定使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

file_name = os.path.join(root_dir, 'model_data','model.h5')

# 超参数
num_classes = 2
batch_size = 64
epochs = 30
dropout_rate = 0.25
reg = regularizers.l1(1e-4)
test_split = 0.2
lr = 1e-4

# 数据增强超参数
horizontal_flip = True
vertical_flip = True
rotation_angle = 180
width_shift_range = 0.1
height_shift_range = 0.1


def build_model():
    base_model = InceptionV3(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D(name='GAP')(x)    #全局平均池化
    x = Dropout(rate=dropout_rate)(x)

    x = Dense(256, activation='elu', name='FC1',kernel_regularizer=reg)(x)
    x = Dropout(rate=dropout_rate)(x)

    x = Dense(128, activation='elu',name='FC2', kernel_regularizer=reg)(x)
    x = Dropout(rate=dropout_rate)(x)

    outputs = Dense(num_classes, activation='softmax',name='Pre')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer=Adam(lr = lr), loss='categorical_crossentropy', metrics=['acc', ])
    model.summary()  # 打印网络结构
    return model


def train_model(model, X_train, y_train, X_valid, y_valid):

    tensorboard = TensorBoard(log_dir='./log', write_graph=False,
                 write_grads=True,
                 write_images=True)

    change_lr = ReduceLROnPlateau(monitor='val_loss',
                      factor=0.25,
                      patience=2,
                      verbose=1,
                      mode='auto',
                      min_lr=1e-7)
    checkpoint = ModelCheckpoint(filepath=file_name, monitor='val_acc', mode='auto', save_best_only='True')

    callback_lists = [tensorboard, change_lr, checkpoint]

    datagen = ImageDataGenerator(rotation_range=rotation_angle,
                                 horizontal_flip=horizontal_flip,
                                 vertical_flip=vertical_flip,
                                 width_shift_range=width_shift_range,
                                 height_shift_range=height_shift_range,
                                 )

    model.fit_generator(generator=datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=X_train.shape[0] // batch_size * 2,
                        epochs=epochs,
                        initial_epoch=0,  # 为啥要有这个参数
                        verbose=1,
                        validation_data=(X_valid, y_valid),
                        callbacks=callback_lists
                        )

if __name__ == '__main__':
    # 加载数据
    X_train, X_valid, y_train, y_valid = load_train_valid_data(test_split)

    # 建立模型
    model = build_model()

    # 训练模型
    train_model(model, X_train, y_train, X_valid, y_valid)


