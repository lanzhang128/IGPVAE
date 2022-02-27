import os
import random
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from utils import load_data, load_model


def perplexity(model, dataset, sample_number=100):
    @tf.function
    def test_step(x):
        kl_loss, rec_loss, elbo = model(x)[1:4]

        return tf.reshape(elbo, shape=(x_batch.shape[0], 1))

    total_ppl = 0
    for step, x_batch in enumerate(dataset):
        sen_len = tf.math.count_nonzero(x_batch, axis=1)
        elbo = tf.zeros(shape=(x_batch.shape[0], 0))
        for _ in range(0, sample_number):
            elbo = tf.keras.backend.concatenate([elbo, test_step(x_batch)], axis=-1)
        log_prob = tf.math.reduce_logsumexp(- elbo, axis=-1) - tf.math.log(sample_number * tf.ones(shape=x_batch.shape[0]))
        batch_ppl = tf.keras.backend.exp(-log_prob / tf.keras.backend.cast_to_floatx(sen_len))
        total_ppl = total_ppl + tf.keras.backend.mean(batch_ppl)
    return total_ppl / (step + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model evaluation')
    parser.add_argument('-s', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-m', '--mpath', default='test', help='path of model')

    args = parser.parse_args()
    seed = args.seed
    model_path = os.path.join(os.path.join(os.getcwd(), 'model'), args.mpath)

    df = pd.read_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'basic.txt'))


    model, batch_size, datapath = load_model(model_path)

    dic = {'Model': os.path.basename(model_path)}
    _, _, test_dataset = load_data(batch_size, datapath, is_train=False)

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dic['PPL'] = float(perplexity(model, test_dataset))
    print(dic)
    df = pd.concat([df, pd.DataFrame(dic, index=[0])], ignore_index=True)
    df.to_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'basic.txt'), index=False, float_format='%.6f')