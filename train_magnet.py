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


def custom_loss(y_true, y_pred):
    custom_loss = (kb.sum(kb.square(
        y_true - y_pred))) / 2
    return custom_loss


def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


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
                    depth=4)
    model.summary()

    if args.plot_model:
        from tensorflow.python.keras.utils.vis_utils import plot_model

        plot_model(model, show_shapes=True)

    dataset = np.load('dataset/scaled_transformer_256.npz')

    trainer = Trainer(name='ResUNet-transformer-original',
                      checkpoint_callback=True)
    x_train = dataset['x_train']
    y_train = dataset['y_train']
    x_valid = dataset['x_valid']
    y_valid = dataset['y_valid']
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
