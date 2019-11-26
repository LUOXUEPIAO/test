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

def Img(imgs):
    lenth = len(imgs)
    images = np.empty((lenth, 224,224,1),dtype="float32")
    count = 0
    for i in imgs:
        img = cv2.imread(i,0)
        img = cv2.resize(img, (224, 224))
        #img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if img.shape == 4:
            img = img[:, :, :3]
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        img = np.array(img)
        img = np.expand_dims(img, axis=2)
        images[count, :, :, :] = img
        count = count+1
    return images
train_images = Img(train_imgs)
lables = np.array(train_lab)
lables = get_dummies(lables)
train_lables = lables.values
test_imgs,test_lab = get_files("D:/test/")

test_images = Img(test_imgs)
tlables = np.array(test_lab)
tlables = get_dummies(tlables)
test_lables = tlables.values

val_indices = np.random.choice(len(test_images),round(len(test_images)*0.5),replace=True)
val_images = test_images[val_indices]
val_lables = test_lables[val_indices]
print(train_images.shape)
print(train_lables.shape)
print(test_images.shape)
print(test_lables.shape)
print(val_images.shape)
print(val_lables.shape)
