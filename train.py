from simple_dataset import SimpleDataset
from resunet.model import ResUNet

if __name__ == "__main__":

    import argparse
    import tensorflow as tf
    import numpy as np
    import random

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dataset_dir_path", required=True, type=str,
                        help="Path to the training dataset. Expects 'images' and 'masks' directory inside with images and masks named the same.")
    parser.add_argument("--validation_dataset_dir_path", required=True,
                        type=str,
                        help="Path to the validation dataset. Expects 'images' and 'masks' directory inside with images and masks named the same.")
    parser.add_argument("--logs_root", default="logs", type=str)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--plot_model", action="store_true", default=False)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    model = ResUNet(input_shape=(128, 128, 1), classes=2, filters_root=16,
                    depth=3)
    model.summary()

    if args.plot_model:
        from tensorflow.python.keras.utils.vis_utils import plot_model

        plot_model(model, show_shapes=True)

    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["categorical_accuracy"])

    train_dataset = list(
        zip(*list(SimpleDataset(args.train_dataset_dir_path)())))
    train_dataset = (np.array(train_dataset[0]), np.array(train_dataset[1]))
    x = np.array(train_dataset[0])
    y = np.array(train_dataset[1])

    validation_dataset = list(
        zip(*list(SimpleDataset(args.validation_dataset_dir_path)())))
    validation_dataset = (
    np.array(validation_dataset[0]), np.array(validation_dataset[1]))
    model.fit(x=x, y=y, validation_data=validation_dataset, epochs=args.epochs,
              batch_size=args.batch_size)
