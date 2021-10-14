import tensorflow as tf
import numpy as np
import random
import argparse
import os
import time
import pandas as pd
from utils import load_data
from lstm_vae import LSTMVAE


def train(model, optimizer, epochs, ckpt_man, ckpt_dir, train_dataset, val_dataset, beta, C_target, C_step=5000):
    @tf.function
    def train_step(x, C_value):
        with tf.GradientTape() as tape:
            kl_loss, rec_loss, elbo = model(x)[1:4]

            loss = tf.keras.backend.mean(tf.keras.backend.abs(kl_loss - C_value) * beta + rec_loss)

        grads = tape.gradient(loss, model.weights)
        optimizer.apply_gradients(zip(grads, model.weights))

        elbo = tf.keras.backend.mean(elbo)
        kl_loss = tf.keras.backend.mean(kl_loss)
        rec_loss = tf.keras.backend.mean(rec_loss)
        return kl_loss, rec_loss, elbo, loss

    @tf.function
    def test_step(x):
        kl_loss, rec_loss, elbo = model(x)[1:4]

        loss = tf.keras.backend.mean(tf.keras.backend.abs(kl_loss - C_target) * beta + rec_loss)
        elbo = tf.keras.backend.mean(elbo)
        kl_loss = tf.keras.backend.mean(kl_loss)
        rec_loss = tf.keras.backend.mean(rec_loss)
        return kl_loss, rec_loss, elbo, loss

    if epochs <= 0:
        ckpt_man.save()

    # please refer to https://keras.io/guides/writing_a_training_loop_from_scratch/
    step_count = 1
    dic = {'step': [], 'loss': [], 'kl_loss': [], 'rec_loss': [], 'elbo': []}

    print('loss=beta({:f})*|kl-C({:f})|+rec)'.format(beta, C_target))

    for epoch in range(1, epochs + 1):
        print("Start of epoch {:d}".format(epoch))
        start_time = time.time()

        total_loss = 0
        total_kl_loss = 0
        total_rec_loss = 0
        total_elbo = 0
        for step, x_batch_train in enumerate(train_dataset):
            if step_count <= C_step and C_target > 0:
                C_value = C_target * step_count / C_step
            else:
                C_value = C_target
            kl_loss, rec_loss, elbo, loss = train_step(x_batch_train, tf.constant(C_value * 1.0))

            total_loss = total_loss + loss
            total_elbo = total_elbo + elbo
            total_rec_loss = total_rec_loss + rec_loss
            total_kl_loss = total_kl_loss + kl_loss

            dic['step'].append(step_count)
            dic['loss'].append(loss)
            dic['kl_loss'].append(kl_loss)
            dic['rec_loss'].append(rec_loss)
            dic['elbo'].append(elbo)

            if step_count % 100 == 0:
                print("step:{:d} train_loss:{:.4f} train_kl_loss:{:.4f} train_rec_loss:{:.4f} train_elbo:{:.4f}"
                      .format(step_count, loss, kl_loss, rec_loss, elbo))

            step_count = step_count + 1

        train_loss, train_kl_loss, train_rec_loss, train_elbo = \
            total_loss / (step + 1), total_kl_loss / (step + 1), total_rec_loss / (step + 1), total_elbo / (step + 1)
        with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
            f.write("train_loss:{:.4f} train_kl_loss:{:.4f} train_rec_loss:{:.4f} train_elbo:{:.4f} ".format(
                train_loss, train_kl_loss, train_rec_loss, train_elbo))
        print("train_loss:{:.4f} train_kl_loss:{:.4f} train_rec_loss:{:.4f} train_elbo:{:.4f}".format(
            train_loss, train_kl_loss, train_rec_loss, train_elbo))

        total_loss = 0
        total_kl_loss = 0
        total_rec_loss = 0
        total_elbo = 0
        for step, x_batch_val in enumerate(val_dataset):
            kl_loss, rec_loss, elbo, loss = test_step(x_batch_val)

            total_loss = total_loss + loss
            total_elbo = total_elbo + elbo
            total_rec_loss = total_rec_loss + rec_loss
            total_kl_loss = total_kl_loss + kl_loss

        val_loss, val_kl_loss, val_rec_loss, val_elbo = \
            total_loss / (step + 1), total_kl_loss / (step + 1), total_rec_loss / (step + 1), total_elbo / (step + 1)
        with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
            f.write("loss:{:.4f} kl_loss:{:.4f} rec_loss:{:.4f} elbo:{:.4f}\n".format(
                val_loss, val_kl_loss, val_rec_loss, val_elbo))
        print("loss:{:.4f} kl_loss:{:.4f} rec_loss:{:.4f} elbo:{:.4f}".format(
            val_loss, val_kl_loss, val_rec_loss, val_elbo))

        ckpt_man.save()
        print("time taken:{:.2f}s".format(time.time() - start_time))

    print('training ends, model at {}'.format(ckpt_dir))
    df = pd.DataFrame(dic)
    df.to_csv(os.path.join(ckpt_dir, 'train_loss.txt'), index=False, float_format='%.6f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ablation study')
    parser.add_argument('-e', '--emb_dim', default=50, type=int, help='embedding dimensions, default: 50')
    parser.add_argument('-r', '--rnn_dim', default=128, type=int, help='RNN dimensions, default: 128')
    parser.add_argument('-z', '--z_dim', default=8, type=int, help='latent space dimensions, default: 8')
    parser.add_argument('-b', '--batch', default=100, type=int, help='batch size, default: 100')
    parser.add_argument('-lr', '--learning_rate', default=0.0005, type=float, help='learning rate, default: 0.0005')
    parser.add_argument('--datapath', default='snli', help='path of data under dataset directory, default: snli')
    parser.add_argument('-pr', '--prior', default='ig', help='prior, default: Isotropic Gaussian')
    parser.add_argument('-s', '--seed', default=0, type=int, help='global random seed')

    args = parser.parse_args()

    seed = args.seed
    emb_dim = args.emb_dim
    rnn_dim = args.rnn_dim
    z_dim = args.z_dim
    batch_size = args.batch
    prior = args.prior
    lr = args.learning_rate
    datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), args.datapath)

    # https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    word2index, index2word, train_dataset, val_dataset, test_dataset = load_data(batch_size, datapath, is_train=True)

    # base igp model
    ckpt_dir = os.path.join(os.path.join(os.getcwd(), 'model'), 'snli_igp_base_' + str(seed))

    if os.system('mkdir ' + ckpt_dir) != 0:
        print('This is not first training.')
        exit()

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    base_model = LSTMVAE(emb_dim=emb_dim, rnn_dim=rnn_dim, z_dim=z_dim, vocab_size=len(word2index),
                         prior=prior, post='iso')
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=base_model)
    ckpt_man = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, max_to_keep=1, checkpoint_name='ckpt')

    with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
        f.write("training configure: prior {}, posterior {}, embedding dimension {:d}, RNN dimension {:d}, "
                "z dimension {:d}, batch size {:d}, epoch number {:d}, learning rate {:f}, "
                "beta {:f}, C {:f}, dataset {}, vocabulary size {:d}\n"
                .format(prior, 'iso', emb_dim, rnn_dim, z_dim, batch_size, 5, lr, 1.0, 5.0, datapath, len(word2index)))

    train(base_model, optimizer, 5, ckpt_man, ckpt_dir, train_dataset, val_dataset, 1.0, 5.0)

    # igp model with C=5
    ckpt_dir = os.path.join(os.path.join(os.getcwd(), 'model'), 'snli_igp_C_5_' + str(seed))

    if os.system('mkdir ' + ckpt_dir) != 0:
        print('This is not first training.')
        exit()

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    igp_model = LSTMVAE(emb_dim=emb_dim, rnn_dim=rnn_dim, z_dim=z_dim, vocab_size=len(word2index),
                        prior=prior, post='iso')

    for step, x_batch_val in enumerate(val_dataset):
        igp_model(x_batch_val)
        break

    igp_model.embeddings.set_weights(base_model.embeddings.get_weights())
    igp_model.encoder_rnn.set_weights(base_model.encoder_rnn.get_weights())
    igp_model.decoder_rnn.set_weights(base_model.decoder_rnn.get_weights())
    igp_model.decoder_vocab_prob.set_weights(base_model.decoder_vocab_prob.get_weights())
    igp_model.encoder_mean_layer.set_weights(base_model.encoder_mean_layer.get_weights())
    igp_model.encoder_logvar_layer.set_weights(base_model.encoder_logvar_layer.get_weights())
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=igp_model)
    ckpt_man = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, max_to_keep=1, checkpoint_name='ckpt')

    with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
        f.write("training configure: prior {}, posterior {}, embedding dimension {:d}, RNN dimension {:d}, "
                "z dimension {:d}, batch size {:d}, epoch number {:d}, learning rate {:f}, "
                "beta {:f}, C {:f}, dataset {}, vocabulary size {:d}\n"
                .format(prior, 'iso', emb_dim, rnn_dim, z_dim, batch_size, 10, lr, 1.0, 5.0, datapath, len(word2index)))

    train(igp_model, optimizer, 10, ckpt_man, ckpt_dir, train_dataset, val_dataset, 1.0, 5.0, C_step=0)

    # dgp model with vanilla ELBO
    ckpt_dir = os.path.join(os.path.join(os.getcwd(), 'model'), 'snli_dgp_elbo_' + str(seed))

    if os.system('mkdir ' + ckpt_dir) != 0:
        print('This is not first training.')
        exit()

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    dgp_model = LSTMVAE(emb_dim=emb_dim, rnn_dim=rnn_dim, z_dim=z_dim, vocab_size=len(word2index),
                        prior=prior, post='diag')

    for step, x_batch_val in enumerate(val_dataset):
        dgp_model(x_batch_val)
        break

    dgp_model.embeddings.set_weights(base_model.embeddings.get_weights())
    dgp_model.encoder_rnn.set_weights(base_model.encoder_rnn.get_weights())
    dgp_model.decoder_rnn.set_weights(base_model.decoder_rnn.get_weights())
    dgp_model.decoder_vocab_prob.set_weights(base_model.decoder_vocab_prob.get_weights())
    dgp_model.encoder_mean_layer.set_weights(base_model.encoder_mean_layer.get_weights())
    igp_logvar_weight = base_model.encoder_logvar_layer.get_weights()
    dgp_logvar_weight = [np.repeat(igp_logvar_weight[i], z_dim, axis=-1) for i in range(len(igp_logvar_weight))]
    dgp_model.encoder_logvar_layer.set_weights(dgp_logvar_weight)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=dgp_model)
    ckpt_man = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, max_to_keep=1, checkpoint_name='ckpt')

    with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
        f.write("training configure: prior {}, posterior {}, embedding dimension {:d}, RNN dimension {:d}, "
                "z dimension {:d}, batch size {:d}, epoch number {:d}, learning rate {:f}, "
                "beta {:f}, C {:f}, dataset {}, vocabulary size {:d}\n"
                .format(prior, 'diag', emb_dim, rnn_dim, z_dim, batch_size, 10, lr, 1.0, 0.0, datapath,
                        len(word2index)))

    train(dgp_model, optimizer, 10, ckpt_man, ckpt_dir, train_dataset, val_dataset, 1.0, 0.0)

    # dgp model with C=5
    ckpt_dir = os.path.join(os.path.join(os.getcwd(), 'model'), 'snli_dgp_C_5_' + str(seed))

    if os.system('mkdir ' + ckpt_dir) != 0:
        print('This is not first training.')
        exit()

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    dgp_model = LSTMVAE(emb_dim=emb_dim, rnn_dim=rnn_dim, z_dim=z_dim, vocab_size=len(word2index),
                        prior=prior, post='diag')

    for step, x_batch_val in enumerate(val_dataset):
        dgp_model(x_batch_val)
        break

    dgp_model.embeddings.set_weights(base_model.embeddings.get_weights())
    dgp_model.encoder_rnn.set_weights(base_model.encoder_rnn.get_weights())
    dgp_model.decoder_rnn.set_weights(base_model.decoder_rnn.get_weights())
    dgp_model.decoder_vocab_prob.set_weights(base_model.decoder_vocab_prob.get_weights())
    dgp_model.encoder_mean_layer.set_weights(base_model.encoder_mean_layer.get_weights())
    igp_logvar_weight = base_model.encoder_logvar_layer.get_weights()
    dgp_logvar_weight = [np.repeat(igp_logvar_weight[i], z_dim, axis=-1) for i in range(len(igp_logvar_weight))]
    dgp_model.encoder_logvar_layer.set_weights(dgp_logvar_weight)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=dgp_model)
    ckpt_man = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, max_to_keep=1, checkpoint_name='ckpt')

    with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
        f.write("training configure: prior {}, posterior {}, embedding dimension {:d}, RNN dimension {:d}, "
                "z dimension {:d}, batch size {:d}, epoch number {:d}, learning rate {:f}, "
                "beta {:f}, C {:f}, dataset {}, vocabulary size {:d}\n"
                .format(prior, 'diag', emb_dim, rnn_dim, z_dim, batch_size, 10, lr, 1.0, 5.0, datapath,
                        len(word2index)))

    train(dgp_model, optimizer, 10, ckpt_man, ckpt_dir, train_dataset, val_dataset, 1.0, 5.0, C_step=0)

    # dgp model with beta=0.4
    ckpt_dir = os.path.join(os.path.join(os.getcwd(), 'model'), 'snli_dgp_beta_0.4_' + str(seed))

    if os.system('mkdir ' + ckpt_dir) != 0:
        print('This is not first training.')
        exit()

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    dgp_model = LSTMVAE(emb_dim=emb_dim, rnn_dim=rnn_dim, z_dim=z_dim, vocab_size=len(word2index),
                        prior=prior, post='diag')

    for step, x_batch_val in enumerate(val_dataset):
        dgp_model(x_batch_val)
        break

    dgp_model.embeddings.set_weights(base_model.embeddings.get_weights())
    dgp_model.encoder_rnn.set_weights(base_model.encoder_rnn.get_weights())
    dgp_model.decoder_rnn.set_weights(base_model.decoder_rnn.get_weights())
    dgp_model.decoder_vocab_prob.set_weights(base_model.decoder_vocab_prob.get_weights())
    dgp_model.encoder_mean_layer.set_weights(base_model.encoder_mean_layer.get_weights())
    igp_logvar_weight = base_model.encoder_logvar_layer.get_weights()
    dgp_logvar_weight = [np.repeat(igp_logvar_weight[i], z_dim, axis=-1) for i in range(len(igp_logvar_weight))]
    dgp_model.encoder_logvar_layer.set_weights(dgp_logvar_weight)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=dgp_model)
    ckpt_man = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, max_to_keep=1, checkpoint_name='ckpt')

    with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
        f.write("training configure: prior {}, posterior {}, embedding dimension {:d}, RNN dimension {:d}, "
                "z dimension {:d}, batch size {:d}, epoch number {:d}, learning rate {:f}, "
                "beta {:f}, C {:f}, dataset {}, vocabulary size {:d}\n"
                .format(prior, 'diag', emb_dim, rnn_dim, z_dim, batch_size, 10, lr, 0.4, 0.0, datapath,
                        len(word2index)))

    train(dgp_model, optimizer, 10, ckpt_man, ckpt_dir, train_dataset, val_dataset, 0.4, 0.0)
