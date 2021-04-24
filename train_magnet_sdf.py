import argparse
import random

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from resunet.model import ResUNet
from train_magnet_functions import train
from trainer import Trainer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dataset_dir_path", required=False, type=str,
                        help="Path to the training dataset. Expects 'images' and 'masks' directory inside with images and masks named the same.")
    parser.add_argument("--validation_dataset_dir_path", required=False,
                        type=str,
                        help="Path to the validation dataset. Expects 'images' and 'masks' directory inside with images and masks named the same.")
    parser.add_argument("--logs_root", default="logs", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--plot_model", action="store_true", default=False)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    args.plot_model = True

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    model = ResUNet(input_shape=(256, 256, 2), classes=2, filters_root=16,
                    depth=4,final_activation='sigmoid')
    model.summary()

    if args.plot_model:
        from tensorflow.python.keras.utils.vis_utils import plot_model

        plot_model(model, show_shapes=True)
    print(args)
    dataset = np.load('c:/datasets/transformer/sdf.npy')
    x = dataset[:, :, :, :-1]
    x[:, :, :, 1] = x[:, :, :, 1] * x[:, :, :, 0]
    y = dataset[:, :, :, -1]

    # scale
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    n, row, col, channel = x.shape
    x_f = x.reshape(n * row * col, channel)
    n, row, col = y.shape
    y_f = y.reshape(n * row * col, 1)
    x_scaler.fit(x_f)
    y_scaler.fit(y_f)
    x_f = x_scaler.transform(x_f)
    y_f = y_scaler.transform(y_f)
    y = y_f.reshape(n, row, col, 1)
    x = x_f.reshape(n, row, col, channel)
    # split
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.33, random_state=42)
    y_train = np.concatenate((y_train, 1 - y_train), axis=3)
    y_valid = np.concatenate((y_valid, 1 - y_valid), axis=3)
    del x_f, y_f, x, y, dataset
    trainer = Trainer(name='ResUNet-transformer-sdf',
                      checkpoint_callback=True)
    train(
        model=model,
        trainer=trainer,
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid
    )