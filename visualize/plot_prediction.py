import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from visualize import utils

prediction=np.load('../pred_200epoch_original.npy')
dataset = np.load('../dataset/scaled_transformer_256.npz')
x_valid = dataset['x_valid']
y_valid = dataset['y_valid']
valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid,y_valid))
scaler=np.load('../dataset/scaled_transformer_256.npz')
with plt.style.context(['science', 'ieee']):
    n_dataset=3
    fig, ax = plt.subplots(n_dataset, 3, sharex=True, sharey=True)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.5)
    for i, (image, label) in enumerate(valid_dataset.take(n_dataset)):
        gt = label[..., 0]
        pred = prediction[i, ...][..., 0]
        pred2 = prediction[i, ...][..., 1]

        error = np.divide(abs(gt - pred), 1, out=np.zeros_like(abs(gt - pred)), where=gt != 0)
        print(pred.max())
        print(gt.numpy().max())
        #     ax1=ax[i][0].matshow(image[..., -1],cmap=plt.cm.jet)
        a1 = ax[i][0].matshow(gt, cmap='jet', norm=norm, aspect='auto')
        ax[i][0].axis('off')
        ax[i][0].set_aspect('equal', 'box')
        a2 = ax[i][1].matshow(pred, cmap='jet', norm=norm, aspect='auto')
        ax[i][1].axis('off')
        ax[i][1].set_aspect('equal', 'box')
        #     a2=ax[i][2].matshow(pred2, cmap=plt.cm.jet,norm=norm, aspect='auto')
        #     error2[error2]=0
        a3 = ax[i][2].matshow(error, cmap='jet', vmin=0,vmax=1, aspect='auto')
        ax[i][2].axis('off')
        ax[i][2].set_aspect('equal', 'box')
        if i == 0:
            ax[i][0].title.set_text(r'$B_{FEM}$')
            ax[i][0].title.set_size(8)
            ax[i][1].title.set_text(r'$B_{DNN}$')
            ax[i][1].title.set_size(8)
            ax[i][2].title.set_text(r'$B_{Error}$')
            ax[i][2].title.set_size(8)
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    # 在原fig上添加一个子图句柄为cbar_ax, 设置其位置为[0.85,0.15,0.05,0.7]
    # colorbar 左 下 宽 高
    l = 0.85
    b = 0.15
    w = 0.05
    h = 1 - 2 * b
    # 对应 l,b,w,h；设置colorbar位置；
    rect = [l, b, w, h]
    cbar_ax = fig.add_axes(rect)

    cb = fig.colorbar(a1, cax=cbar_ax)
    fig.savefig('aa.png')