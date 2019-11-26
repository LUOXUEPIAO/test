import numpy as np
import os
from pandas import get_dummies
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
from PIL import  Image

def get_files(file_path):
    class_train = []
    label_train = []
    for train_class in os.listdir(file_path):
        for pic_name in os.listdir(file_path + train_class):
            class_train.append(file_path + train_class + '/' + pic_name)
            label_train.append(train_class)
    temp = np.array([class_train, label_train])
    temp = temp.transpose()

    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])

    label_list = [int(i) for i in label_list]
    return image_list, label_list



train_imgs,train_lab = get_files("D:/garbage_classify/")

def Img():
    images = np.empty((14000,224,224,1),dtype="float32")
    count=0
    for i in train_imgs:
        if count >= 14000:
            break
        img = cv2.imread(i,0)
        img = cv2.resize(img, (224, 224 ))
        #img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if img.shape == 4:
            img = img[:, :, :3]
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        img = np.array(img)
        img = np.expand_dims(img, axis=2)
        images[count, :, :, :] = img
        count=count+1
    return images
train_images = Img()
lables = np.array(train_lab)
lables = get_dummies(lables)
train_lables = lables.values
print(train_images)


