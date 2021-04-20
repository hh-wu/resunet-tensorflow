from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler

dataset = np.load('c:/datasets/transformer/dataset_mat.npy')
# Split the data
# x_train, x_valid, y_train, y_valid = train_test_split(X_edf, Y,
#                                                       test_size=0.33,
#                                                       shuffle=True)

# 将dataset搞成256*256大小，之前大小是3375*237*253*4
dataset_256_256 = np.pad(dataset, ((0, 0), (0, 0), (3, 0), (0, 0)), 'constant',
                         constant_values=0)
dataset_256_256 = dataset_256_256[:, :-1, :, :]

datasize_train, row, col, channel = dataset_256_256.shape
material = dataset_256_256[:, :, :, 0].reshape(datasize_train * row * col)
# edf=dataset_with_edf2[:,:,:,1].reshape(datasize_train*row*col)
j = dataset_256_256[:, :, :, 1].reshape(datasize_train * row * col)
bx = dataset_256_256[:, :, :, 2].reshape(datasize_train * row * col)
by = dataset_256_256[:, :, :, 3].reshape(datasize_train * row * col)
bmag = np.sqrt(dataset_256_256[:, :, :, 3] ** 2 + dataset_256_256[:, :, :,
                                                  2] ** 2).reshape(
    datasize_train * row * col)

All_data_edf = np.zeros((datasize_train * row * col, 3))
All_data_edf[:, 0] = material
All_data_edf[:, 1] = j
All_data_edf[:, 2] = bmag

del material, j, bx, by, bmag
scaler_edf = MinMaxScaler()
scaler_edf.fit(All_data_edf)

All_data_scaled_edf = scaler_edf.transform(All_data_edf)

X = np.zeros((datasize_train, row, col, 2))
Y = np.zeros((datasize_train, row, col, 2))
X[:, :, :, 0] = All_data_scaled_edf[:, 0].reshape(datasize_train, row, col)
X[:, :, :, 1] = All_data_scaled_edf[:, 1].reshape(datasize_train, row, col)
Y[:, :, :, 0] = All_data_scaled_edf[:, 2].reshape(datasize_train, row, col)
Y[:, :, :, 1] = 1 - All_data_scaled_edf[:, 2].reshape(datasize_train, row, col)

from sklearn.model_selection import train_test_split

# Split the data
x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.33,
                                                      shuffle=True)

np.savez('dataset/scaled_transformer_256.npz',
        x_train=x_train, x_valid=x_valid,
        y_train=y_train, y_valid=y_valid)
