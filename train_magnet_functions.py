from sklearn.preprocessing import MinMaxScaler

from resunet.model import ResUNet
import argparse
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow.keras.backend as kb
from tensorflow.keras import losses
from sklearn.model_selection import train_test_split

from simple_dataset import SimpleDataset
from trainer import Trainer


def minmax(mat: np.ndarray):
    for i in range(mat.shape[3]):
        print(
            f'layer{i + 1}: '
            f'min={mat[:, :, :, i].min():12.2f},'
            f'max={mat[:, :, :, i].max():12.2f}.')


def custom_loss(y_true, y_pred):
    custom_loss = (kb.sum(kb.square(
        y_true - y_pred))) / 2
    return custom_loss


def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def train(model, trainer, x_train, y_train, x_valid, y_valid):
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train))
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (x_valid, y_valid))

    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["categorical_accuracy", ssim_loss, custom_loss])
    trainer.fit(model=model,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                epochs=10,
                batch_size=32)

    model.compile(loss=custom_loss, optimizer="adam",
                  metrics=["categorical_accuracy", ssim_loss, custom_loss])
    trainer.fit(model=model,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                epochs=300,
                batch_size=32)


def get_dataset(full_dataset, excitation: str = 'original'):
    material=np.copy(full_dataset[:, :, :, 0])
    material[material<=2]=1000
    material[material<23]=1
    j = full_dataset[:, :, :, 1]
    bx = full_dataset[:, :, :, 2]
    by = full_dataset[:, :, :, 3]
    edf = full_dataset[:, :, :, 4]
    sdf = full_dataset[:, :, :, 5]
    bmag = np.sqrt(bx ** 2 + by ** 2)
    if excitation == 'original':
        return np.stack((material, j, bmag), axis=-1)
    elif excitation == 'sdf':
        return np.stack((material, sdf, bmag), axis=-1)
    elif excitation == 'edf':
        return np.stack((material, edf, bmag), axis=-1)

def plot(dataset,layers=3):
    # for i in range(layers):
    plt.subplot(131)
    plt.imshow(dataset[-1,:,:,0])
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(dataset[-1,:,:,1])
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(dataset[-1,:,:,2])
    plt.colorbar()
    plt.show()

def scale(dataset):
    scalar=MinMaxScaler()
    scalar.fit()
    return