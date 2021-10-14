import os
import random
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from utils import load_data, load_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model evaluation')
    parser.add_argument('-s', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-m', '--mpath', default='test', help='path of model')

    args = parser.parse_args()
    seed = args.seed
    model_path = args.mpath

    if not os.path.exists(os.path.join(os.path.join(os.getcwd(), 'model'), 'aggregated.txt')):
        df = pd.DataFrame(columns=['Model', 'logdetcov', 'mean_square'])
        df.to_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'aggregated.txt'), index=False, float_format='%.6f')
    df = pd.read_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'aggregated.txt'))

    model, batch_size, datapath = load_model(model_path)

    if os.path.basename(model_path) in list(df['Model']):
        print('Results already exists.')
        exit()
    else:
        dic = {'Model': os.path.basename(model_path)}
        _, _, test_dataset = load_data(batch_size, datapath, is_train=False)

        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        logdetcov = []
        mean_square = []
        for _ in range(5):
            initial = True
            for x_batch_test in test_dataset:
                if initial:
                    sample_z = model.encoding(x_batch_test)[0]
                    initial = False
                else:
                    z = model.encoding(x_batch_test)[0]
                    sample_z = tf.keras.backend.concatenate([sample_z, z], axis=0)
            logdetcov.append(np.log(np.linalg.det(np.cov(sample_z.numpy(), rowvar=False))))
            mean_square.append(np.sum(np.square(np.mean(sample_z.numpy(), axis=0))))

        dic['logdetcov'] = np.mean(logdetcov)
        dic['mean_square'] = np.mean(mean_square)
        print(dic)
        df = pd.concat([df, pd.DataFrame(dic, index=[0])], ignore_index=True)
        df.to_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'aggregated.txt'), index=False, float_format='%.6f')