from keras.layers import Conv2D, MaxPool2D, Dropout, BatchNormalization, Flatten, Dense, UpSampling2D
from keras.layers import Layer
import tensorflow as tf
from Lib import ROIPoolingLayer


# Faster R-CNN implementation
# Generate layer that are base for all the models
def Base(InputLayer):
    Block1_Conv2D = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(InputLayer)
    Block1_Conv2D = Dropout(0.1)(Block1_Conv2D)
    Block1_Conv2D = BatchNormalization()(Block1_Conv2D)
    Block1_Conv2D = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(Block1_Conv2D)
    Block1_Conv2D = Dropout(0.1)(Block1_Conv2D)
    Block1_Conv2D = BatchNormalization()(Block1_Conv2D)
    Block1_Conv2D = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(Block1_Conv2D)

    Block2_Conv2D = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(Block1_Conv2D)
    Block2_Conv2D = Dropout(0.1)(Block2_Conv2D)
    Block2_Conv2D = BatchNormalization()(Block2_Conv2D)
    Block2_Conv2D = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(Block2_Conv2D)
    Block2_Conv2D = Dropout(0.1)(Block2_Conv2D)
    Block2_Conv2D = BatchNormalization()(Block2_Conv2D)
    Block2_Conv2D = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(Block2_Conv2D)

    Block3_Conv2D = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(Block2_Conv2D)
    Block3_Conv2D = Dropout(0.1)(Block3_Conv2D)
    Block3_Conv2D = BatchNormalization()(Block3_Conv2D)
    Block3_Conv2D = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(Block3_Conv2D)
    Block3_Conv2D = Dropout(0.1)(Block3_Conv2D)
    Block3_Conv2D = BatchNormalization()(Block3_Conv2D)
    Block3_Conv2D = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(Block3_Conv2D)
    Block3_Conv2D = Dropout(0.1)(Block3_Conv2D)
    Block3_Conv2D = BatchNormalization()(Block3_Conv2D)
    Block3_Conv2D = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(Block3_Conv2D)

    Block4_Conv2D = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(Block3_Conv2D)
    Block4_Conv2D = Dropout(0.1)(Block4_Conv2D)
    Block4_Conv2D = BatchNormalization()(Block4_Conv2D)
    Block4_Conv2D = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(Block4_Conv2D)
    Block4_Conv2D = Dropout(0.1)(Block4_Conv2D)
    Block4_Conv2D = BatchNormalization()(Block4_Conv2D)
    Block4_Conv2D = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(Block4_Conv2D)
    Block4_Conv2D = Dropout(0.1)(Block4_Conv2D)
    Block4_Conv2D = BatchNormalization()(Block4_Conv2D)
    Block4_Conv2D = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(Block4_Conv2D)

    Block5_Conv2D = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(Block4_Conv2D)
    Block5_Conv2D = Dropout(0.1)(Block5_Conv2D)
    Block5_Conv2D = BatchNormalization()(Block5_Conv2D)
    Block5_Conv2D = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(Block5_Conv2D)
    Block5_Conv2D = Dropout(0.1)(Block5_Conv2D)
    Block5_Conv2D = BatchNormalization()(Block5_Conv2D)
    Block5_Conv2D = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(Block5_Conv2D)
    Block5_Conv2D = Dropout(0.1)(Block5_Conv2D)
    Block5_Conv2D = BatchNormalization()(Block5_Conv2D)
    Block5_Conv2D = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(Block5_Conv2D)

    return Block5_Conv2D


# Region proposal network
def RPN(BaseLayer, AnchorCount):
    L1 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(BaseLayer)
    L1 = Dropout(0.1)(L1)
    L1 = BatchNormalization()(L1)
    Class = Conv2D(filters=AnchorCount, kernel_size=(1, 1), activation='sigmoid')(L1)
    Regressor = Conv2D(filters=AnchorCount * 4, kernel_size=(1, 1), activation='linear')(L1)
    return [Class, Regressor]


# RoI pooling layer
def RoiPooling(BaseLayer, InputROI, PoolingHeight, PoolingWidth):
    return ROIPoolingLayer(PoolingHeight, PoolingWidth)([BaseLayer, InputROI])


# Classifier model
def Classifier(BaseLayer, ClassesCount):
    Out = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(BaseLayer)
    Out = Flatten()(Out)
    Out = Dense(1024, activation="relu")(Out)
    Out = Dropout(0.1)(Out)
    Out = BatchNormalization()(Out)
    Out = Dense(1024, activation="relu")(Out)
    Out = Dropout(0.1)(Out)
    Out = BatchNormalization()(Out)
    Class = Dense(units=ClassesCount, activation='softmax')(Out)
    Regressor = Dense(units=(ClassesCount - 1) * 4, activation='linear')(Out)
    return [Class, Regressor]
