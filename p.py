
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D,BatchNormalization
from tensorflow.keras.optimizers import SGD,Adam,RMSprop
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
IMG_CHANNELS=3
weight_decay = 0.0005
IMG_ROWS=224
IMG_COLS=224
BATCH_SIZE=64
NB_EPOCH=10
NB_CLASSES=2
VERBOSE=1
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_genrendata = data_gen.flow_from_directory(
    'D:/garbage_classify',
    target_size=(224, 224),  # resize图片
    batch_size=8,
    class_mode='binary'
)
test_genrendata = data_gen.flow_from_directory(
    'D:/Download/test',
    target_size=(224, 224),  # resize图片
    batch_size=8,
    class_mode='binary'
)
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(IMG_ROWS, IMG_COLS, 3), kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
# layer14 1*1*512
model.add(Flatten())
model.add(Dense(128, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(40))
model.add(Activation('softmax'))
# 10
model.summary()

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-4), metrics=['accuracy'])
model.fit(train_genrendata,epochs=10, steps_per_epoch=100)
score=model.evaluate(x_test,y_test,batch_size=BATCH_SIZE,verbose=VERBOSE)
print("Test score:",score[0])
print("Test accuracy:",score[1])
model_json=model.to_json()
open('cifar10_architecture.json','w').write(model_json)
#and the weights learnde by out deep network on the training set
model.save_weights('cifar10_weights.h5',overwrite=True)
