import argparse
import random

import numpy as np
import tensorflow as tf

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

    dataset = np.load('dataset/scaled_transformer_256.npz')

    trainer = Trainer(name='ResUNet-transformer-original',
                      checkpoint_callback=True)
    x_train = dataset['x_train']
    y_train = dataset['y_train']
    x_valid = dataset['x_valid']
    y_valid = dataset['y_valid']
    train(
        model=model,
        trainer=trainer,
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid
    )