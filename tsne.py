import os
import argparse
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='classification task')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed')
    parser.add_argument('-m', '--mpath', default='DBpedia\\DBpedia_C_15_pr_ig_po_iso_1', help='path of model')

    args = parser.parse_args()

    seed = args.seed
    model_path = args.mpath

    with open(os.path.join(model_path, 'epoch_loss.txt'), 'r') as f:
        s = f.readlines()[0]
        s = s.split(',')

    datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), os.path.basename(s[-2].split()[-1]))
    x_df = pd.read_csv(os.path.join(model_path, 'representation.csv'), index_col='index')
    x = np.array(x_df)

    labels = []
    with open(os.path.join(datapath, 'test_class_label.txt'), 'r') as f:
        for label in f.readlines():
            label = int(label.rstrip())
            labels.append(label)

    color = list(plt.cm.tab10.colors) + \
            [plt.cm.Dark2.colors[0], plt.cm.Dark2.colors[3], plt.cm.Set1.colors[5], plt.cm.Paired.colors[9]]

    tsne = TSNE(random_state=seed)
    y = tsne.fit_transform(x)
    plt.figure(figsize=(15, 9))
    plt.rcParams['font.family'] = 'Times New Roman'
    for i in range(0, len(list(set(labels)))):
        plt.scatter(y[i*1000:(i+1)*1000, 0], y[i*1000:(i+1)*1000, 1], label=str(i), color=color[i])
    plt.axis('off')
    #plt.gca().invert_xaxis()
    #plt.legend(fontsize=20, markerscale=2.0, loc='upper right')
    plt.tight_layout(pad=0.0)
    plt.savefig(os.path.join(model_path, 'tsne.pdf'))
    plt.close()
