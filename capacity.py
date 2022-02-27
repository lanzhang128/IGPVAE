import tensorflow as tf
import numpy as np
import random
import argparse
import os
import time
import pandas as pd
from utils import load_data
from lstm_vae import LSTMVAE


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capacity Experiment')
    parser.add_argument('-e', '--emb_dim', default=50, type=int, help='embedding dimensions, default: 50')
    parser.add_argument('-r', '--rnn_dim', default=128, type=int, help='RNN dimensions, default: 128')
    parser.add_argument('-z', '--z_dim', default=8, type=int, help='latent space dimensions, default: 8')
    parser.add_argument('-b', '--batch', default=100, type=int, help='batch size, default: 100')
    parser.add_argument('-lr', '--learning_rate', default=0.0005, type=float, help='learning rate, default: 0.0005')
    parser.add_argument('--epochs', default=20, type=int, help='epochs number, default: 20')
    parser.add_argument('--datapath', default='snli', help='path of data under dataset directory, default: snli')
    parser.add_argument('-pr', '--prior', default='ig', help='prior, default: Isotropic Gaussian')
    parser.add_argument('-po', '--posterior', default='diag', help='posterior, default: Diagonal Gaussian')
    parser.add_argument('-beta', default=1, type=float, help='beta for training VAE, default: 1')
    parser.add_argument('-C_step', default=0.1, type=float, help='step of C for training VAE, default: 0.1')
    parser.add_argument('-n', '--num', default=100, type=int, help='the number of steps after which increasing C with '
                                                                   'C_step, default: 100')
    parser.add_argument('-s', '--seed', default=0, type=int, help='global random seed')
    parser.add_argument('-m', '--mpath', default='snli_capacity_diag', help='path of model')

    args = parser.parse_args()

    seed = args.seed
    batch_size = args.batch
    lr = args.learning_rate
    epochs = args.epochs
    datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), args.datapath)
    ckpt_dir = os.path.join(os.path.join(os.getcwd(), 'model'), args.mpath)

    # https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_dataset, val_dataset, test_dataset, word2index, index2word = load_data(batch_size, datapath)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    if os.system('mkdir ' + ckpt_dir) != 0:
        print('This is not first training.')
        exit()
    beta = args.beta
    C_step = args.C_step
    n = args.num
    emb_dim = args.emb_dim
    rnn_dim = args.rnn_dim
    z_dim = args.z_dim
    prior = args.prior
    posterior = args.posterior
    checkpoint_prefix = os.path.join(ckpt_dir, "ckpt")
    model = LSTMVAE(emb_dim=emb_dim, rnn_dim=rnn_dim, z_dim=z_dim, vocab_size=len(word2index), prior=prior, post=posterior)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)

    with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
        f.write(
                "training configure: prior {}, posterior {}, embedding dimension {:d}, RNN dimension {:d}, z dimension {:d}, "
                "batch size {:d}, epoch number {:d}, learning rate {:f}, beta {:f}, C step {:f}, number of iteration: {:d}, "
                "dataset {}, vocabulary size {:d}\n"
                .format(prior, posterior, emb_dim, rnn_dim, z_dim, batch_size, epochs, lr, beta, C_step, n, datapath, len(word2index)))

    @tf.function
    def train_step(x, C_value):
        with tf.GradientTape() as tape:
            kl_loss, rec_loss, vae_loss, mean, logvar = model(x)[1:]

            loss = tf.keras.backend.mean(tf.keras.backend.abs(kl_loss - C_value) * beta + rec_loss)

        grads = tape.gradient(loss, model.weights)
        optimizer.apply_gradients(zip(grads, model.weights))

        vae_loss = tf.keras.backend.mean(vae_loss)
        kl_loss = tf.keras.backend.mean(kl_loss)
        rec_loss = tf.keras.backend.mean(rec_loss)
        return kl_loss, rec_loss, vae_loss, loss, mean, logvar


    @tf.function
    def test_step(x, C_value):
        kl_loss, rec_loss, vae_loss, mean, logvar = model(x)[1:]

        loss = tf.keras.backend.mean(tf.keras.backend.abs(kl_loss - C_value) * beta + rec_loss)
        vae_loss = tf.keras.backend.mean(vae_loss)
        kl_loss = tf.keras.backend.mean(kl_loss)
        rec_loss = tf.keras.backend.mean(rec_loss)
        return kl_loss, rec_loss, vae_loss, loss, mean, logvar


    total_loss = 0
    total_kl_loss = 0
    total_rec_loss = 0
    total_vae_loss = 0
    for step, x_batch_val in enumerate(val_dataset):
        kl_loss, rec_loss, vae_loss, loss = test_step(x_batch_val, tf.constant(0.0))[:4]

        total_loss = total_loss + loss
        total_vae_loss = total_vae_loss + vae_loss
        total_rec_loss = total_rec_loss + rec_loss
        total_kl_loss = total_kl_loss + kl_loss

    val_loss, val_kl_loss, val_rec_loss, val_vae_loss = \
        total_loss / (step + 1), total_kl_loss / (step + 1), total_rec_loss / (step + 1), total_vae_loss / (
                step + 1)
    with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
        f.write("loss:{:.4f} kl_loss:{:.4f} rec_loss:{:.4f} vae_loss:{:.4f}\n".format(
            val_loss, val_kl_loss, val_rec_loss, val_vae_loss))
    print("loss:{:.4f} kl_loss:{:.4f} rec_loss:{:.4f} vae_loss:{:.4f}".format(
        val_loss, val_kl_loss, val_rec_loss, val_vae_loss))

    dim_kl = tf.zeros(shape=(0, z_dim))
    temp_dim_kl = tf.zeros(shape=(0, z_dim))
    # please refer to https://keras.io/guides/writing_a_training_loop_from_scratch/
    steps = np.zeros(shape=(0, 1))
    Cs = np.zeros(shape=(0, 1))
    recs = tf.zeros(shape=(0, 1))
    temp_rec = tf.zeros(shape=(0, 1))
    step_count = 1
    C_value = 0.0

    for epoch in range(1, epochs + 1):
        print("Start of epoch {:d}".format(epoch))
        start_time = time.time()

        total_loss = 0
        total_kl_loss = 0
        total_rec_loss = 0
        total_vae_loss = 0
        for step, x_batch_train in enumerate(train_dataset):
            kl_loss, rec_loss, vae_loss, loss, mean, logvar = train_step(x_batch_train, tf.constant(C_value * 1.0))

            total_loss = total_loss + loss
            total_vae_loss = total_vae_loss + vae_loss
            total_rec_loss = total_rec_loss + rec_loss
            total_kl_loss = total_kl_loss + kl_loss

            kl = 0.5 * (tf.keras.backend.square(mean) + tf.keras.backend.exp(logvar) - 1 - logvar)
            temp_dim_kl = tf.keras.backend.concatenate([temp_dim_kl, kl], axis=0)
            temp_rec = tf.keras.backend.concatenate([temp_rec, rec_loss * tf.ones(shape=(1, 1))], axis=0)

            if step_count % n == 0:
                print("C value:{:f}, step:{:d} train_loss:{:.4f} train_kl_loss:{:.4f} train_rec_loss:{:.4f} "
                      "train_vae_loss:{:.4f}".format(C_value, step_count, loss, kl_loss, rec_loss, vae_loss))
                steps = np.concatenate([steps, step_count * np.ones(shape=(1, 1))], axis=0)
                Cs = np.concatenate([Cs, C_value * np.ones(shape=(1, 1))], axis=0)
                temp_dim_kl = tf.keras.backend.mean(temp_dim_kl, axis=0)
                temp_dim_kl = tf.expand_dims(temp_dim_kl, axis=0)
                dim_kl = tf.keras.backend.concatenate([dim_kl, temp_dim_kl], axis=0)
                recs = tf.keras.backend.concatenate([recs, tf.keras.backend.mean(temp_rec) * tf.ones(shape=(1, 1))], axis=0)
                C_value = C_value + C_step
                temp_dim_kl = tf.zeros(shape=(0, z_dim))
                temp_rec = tf.zeros(shape=(0, 1))

            step_count = step_count + 1

        train_loss, train_kl_loss, train_rec_loss, train_vae_loss = \
            total_loss / (step + 1), total_kl_loss / (step + 1), total_rec_loss / (step + 1), total_vae_loss / (
                        step + 1)
        with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
            f.write("train_loss:{:.4f} train_kl_loss:{:.4f} train_rec_loss:{:.4f} train_vae_loss:{:.4f} ".format(
                train_loss, train_kl_loss, train_rec_loss, train_vae_loss))
        print("train_loss:{:.4f} train_kl_loss:{:.4f} train_rec_loss:{:.4f} train_vae_loss:{:.4f}".format(
            train_loss, train_kl_loss, train_rec_loss, train_vae_loss))

        total_loss = 0
        total_kl_loss = 0
        total_rec_loss = 0
        total_vae_loss = 0
        for step, x_batch_val in enumerate(val_dataset):
            kl_loss, rec_loss, vae_loss, loss, mean, logvar = test_step(x_batch_val, tf.constant(C_value * 1.0))

            total_loss = total_loss + loss
            total_vae_loss = total_vae_loss + vae_loss
            total_rec_loss = total_rec_loss + rec_loss
            total_kl_loss = total_kl_loss + kl_loss

        val_loss, val_kl_loss, val_rec_loss, val_vae_loss = \
            total_loss / (step + 1), total_kl_loss / (step + 1), total_rec_loss / (step + 1), total_vae_loss / (
                        step + 1)
        with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
            f.write("loss:{:.4f} kl_loss:{:.4f} rec_loss:{:.4f} vae_loss:{:.4f}\n".format(
                val_loss, val_kl_loss, val_rec_loss, val_vae_loss))
        print("loss:{:.4f} kl_loss:{:.4f} rec_loss:{:.4f} vae_loss:{:.4f}".format(
            val_loss, val_kl_loss, val_rec_loss, val_vae_loss))
        print("time taken:{:.2f}s".format(time.time() - start_time))
        ckpt.save(file_prefix=checkpoint_prefix)
    print('training ends, model at {}'.format(ckpt_dir))

    recs = recs.numpy()
    dim_kl = dim_kl.numpy()

    capacity_data = np.concatenate([steps, Cs,  recs, dim_kl], axis=-1)
    df = pd.DataFrame(capacity_data)
    df.columns = ['step', 'C', 'rec'] + ['kl'+str(i) for i in range(1, z_dim+1)]
    df.to_csv(os.path.join(ckpt_dir, 'capacity.csv'), index=False)