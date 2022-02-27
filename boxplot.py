import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, load_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model evaluation')
    parser.add_argument('-m', '--mpath', default='test', help='path of model')

    args = parser.parse_args()
    model_path = args.mpath

    dic = {'Model': os.path.basename(model_path)}
    model, batch_size, datapath = load_model(model_path)
    word2index, index2word, test_dataset = load_data(batch_size, datapath, is_train=False)

    initial = True
    for step, x_batch_test in enumerate(test_dataset):
        mean, logvar = model(x_batch_test)[-2:]

        if initial:
            all_mean = mean
            all_var = tf.keras.backend.exp(logvar)
            initial = False
        else:
            all_mean = tf.keras.backend.concatenate([all_mean, mean], axis=0)
            all_var = tf.keras.backend.concatenate([all_var, tf.keras.backend.exp(logvar)], axis=0)
    mean = all_mean.numpy()
    var = all_var.numpy()

    cov = np.cov(mean, rowvar=False)
    au = []
    for i in range(0, mean.shape[1]):
        if cov[i][i] > 0.01:
            au.append(1.0)
        else:
            au.append(0.0)
    au = np.array(au)

    mean, var = mean.T, var.T
    color_plot = ['b', 'r']

    plt.figure(figsize=(8, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 28
    positions = np.array(np.where(au == 1))+1
    positions.resize((positions.shape[1],))
    labels = ['AU' for _ in range(0, positions.shape[0])]
    if len(labels) != 0:
        plt.boxplot(mean[np.where(au==1.0)].T, positions=positions, labels=labels, boxprops={'color':color_plot[0]})
    positions = np.array(np.where(au == 0)) + 1
    positions.resize((positions.shape[1],))
    labels = ['IAU' for _ in range(0, positions.shape[0])]
    if len(labels) != 0:
        plt.boxplot(mean[np.where(au == 0)].T, positions=positions, labels=labels, boxprops={'color': color_plot[1]})
    plt.xlabel('dimension')
    plt.ylabel('Mean Value')
    plt.tick_params(labelsize=20)
    plt.savefig(os.path.join(model_path, 'mean.pdf'))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 28
    positions = np.array(np.where(au == 1)) + 1
    positions.resize((positions.shape[1],))
    labels = ['AU' for _ in range(0, positions.shape[0])]
    if len(labels) != 0:
        plt.boxplot(var[np.where(au == 1.0)].T, positions=positions, labels=labels, boxprops={'color': color_plot[0]})
    positions = np.array(np.where(au == 0)) + 1
    positions.resize((positions.shape[1],))
    labels = ['IAU' for _ in range(0, positions.shape[0])]
    if len(labels) != 0:
        plt.boxplot(var[np.where(au == 0)].T, positions=positions, labels=labels, boxprops={'color': color_plot[1]})
    plt.xlabel('dimension')
    plt.ylim(0, 1.05)
    plt.ylabel('Variance Value')
    plt.tick_params(labelsize=20)
    plt.savefig(os.path.join(model_path, 'var.pdf'))
    plt.close()