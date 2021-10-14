import tensorflow as tf
import numpy as np
import random
import argparse
import time
import os
import pandas as pd
from utils import load_data


class MoGPrior(tf.keras.layers.Layer):
    def __init__(self, dim, comp='diag', num=5):
        super().__init__()
        self.num = num
        self.comp = comp
        self.dim = dim
        self.initializer = tf.keras.initializers.GlorotNormal()
        if self.comp == 'diag':
            self.mean = tf.Variable(initial_value=self.initializer(shape=(self.dim, self.num)), trainable=True)
            self.logvar = tf.Variable(initial_value=self.initializer(shape=(self.dim, self.num)), trainable=True)
        elif self.comp == 'iso':
            self.mean = tf.Variable(initial_value=self.initializer(shape=(self.dim, self.num)), trainable=True)
            self.logvar = tf.Variable(initial_value=self.initializer(shape=(1, self.num)), trainable=True)
        else:
            raise ValueError

    def call(self, sample_number):
        samples = tf.random.normal(shape=(sample_number, self.dim, self.num))
        temp_std = tf.keras.backend.exp(0.5 * self.logvar)
        if self.comp == 'iso':
            temp_std = tf.keras.backend.repeat_elements(temp_std, self.dim, axis=0)
        samples = tf.ones(shape=samples.shape) * self.mean + samples * temp_std
        return tf.keras.backend.mean(samples, axis=-1)

    def log_prior(self, z):
        temp_z = tf.keras.backend.expand_dims(z, axis=-1)
        temp_z = tf.keras.backend.repeat_elements(temp_z, self.num, axis=-1)
        temp_mean = tf.keras.backend.expand_dims(self.mean, axis=0)
        temp_mean = tf.keras.backend.repeat_elements(temp_mean, z.shape[0], axis=0)
        temp_var = tf.keras.backend.exp(self.logvar)
        if self.comp == 'iso':
            temp_var = tf.keras.backend.repeat_elements(temp_var, self.dim, axis=0)
        temp_var = tf.keras.backend.expand_dims(temp_var, axis=0)
        temp_var = tf.keras.backend.repeat_elements(temp_var, z.shape[0], axis=0)
        logits = -0.5 * (tf.keras.backend.log(temp_var) + tf.keras.backend.square(temp_z - temp_mean) / temp_var)
        return tf.math.reduce_logsumexp(tf.keras.backend.sum(logits, axis=1), axis=-1) - tf.math.log(self.num * tf.ones(shape=z.shape[0]))


class LSTMVAE(tf.keras.Model):
    def __init__(self, emb_dim, rnn_dim, z_dim, vocab_size, prior='ig', post='diag'):
        super().__init__()
        self.embeddings = tf.keras.layers.Embedding(vocab_size, emb_dim, mask_zero=True)
        self.encoder_rnn = tf.keras.layers.LSTM(rnn_dim, return_state=False, return_sequences=True,
                                                kernel_initializer='lecun_normal', recurrent_initializer='lecun_normal')
        self.decoder_rnn = tf.keras.layers.LSTM(rnn_dim, return_state=True, return_sequences=True,
                                                kernel_initializer='lecun_normal', recurrent_initializer='lecun_normal')
        self.decoder_vocab_prob = tf.keras.layers.Dense(vocab_size, activation='softmax')
        self.encoder_mean_layer = tf.keras.layers.Dense(z_dim)
        self.prior = prior
        self.post = post
        self.z_dim = z_dim
        if self.post == 'diag':
            self.encoder_logvar_layer = tf.keras.layers.Dense(z_dim)
        elif self.post == 'iso':
            self.encoder_logvar_layer = tf.keras.layers.Dense(1)
        else:
            raise ValueError
        if self.prior == 'mog':
            self.MoG_prior = MoGPrior(z_dim, comp='diag')
        elif self.prior == 'moig':
            self.MoG_prior = MoGPrior(z_dim, comp='iso')
        elif self.prior != 'ig':
            raise ValueError

    def encoding(self, x):
        enc_embeddings = self.embeddings(x)
        mask = self.embeddings.compute_mask(x)
        output = self.encoder_rnn(enc_embeddings, mask=mask)

        # extract the whole sentence representation, 2 is the index of <eos>
        temp_mask = tf.keras.backend.cast_to_floatx(tf.keras.backend.equal(x, 2))
        temp_mask = tf.keras.backend.expand_dims(temp_mask)
        temp_mask = tf.keras.backend.repeat_elements(temp_mask, output.shape[2], axis=2)
        output = tf.keras.backend.sum(output*temp_mask, axis=1)

        # get latent code with diagonal Gaussian
        mean = self.encoder_mean_layer(output)
        if self.post == 'diag':
            logvar = self.encoder_logvar_layer(output)
        else:
            logvar = self.encoder_logvar_layer(output)
            logvar = tf.keras.backend.repeat_elements(logvar, mean.shape[1], axis=1)
        epsilon = tf.random.normal(shape=mean.shape)
        z = mean + tf.keras.backend.exp(0.5 * logvar) * epsilon

        return z, mean, logvar

    def decoder_training(self, x, z):
        # 1 is the index of <bos>
        y = tf.keras.backend.concatenate([tf.constant(1, shape=(x.shape[0], 1)), x[:, :-1]], axis=-1)
        dec_embeddings = self.embeddings(y)
        mask = self.embeddings.compute_mask(y)
        new_z = tf.keras.backend.repeat(z, dec_embeddings.shape[1])
        dec_input = tf.keras.layers.concatenate([dec_embeddings, new_z], axis=-1)
        output, _, _ = self.decoder_rnn(dec_input, mask=mask)
        predictions = self.decoder_vocab_prob(output)
        return predictions

    def kld_loss(self, z, mean, logvar):
        if self.prior == 'ig':
            kld = 0.5 * tf.keras.backend.sum(tf.keras.backend.square(mean) + tf.keras.backend.exp(logvar) - 1 - logvar, axis=-1)
        else:
            post_kld = tf.keras.backend.sum(-0.5 * (logvar + tf.keras.backend.square(z - mean) / tf.keras.backend.exp(logvar)), axis=-1)
            prior_kld = self.MoG_prior.log_prior(z)
            kld = post_kld - prior_kld
        return kld

    @staticmethod
    def reconstruction_loss(x, predictions):
        # ignore padding
        temp_mask = 1 - tf.keras.backend.cast_to_floatx(tf.keras.backend.equal(x, 0))
        prob = tf.keras.backend.sparse_categorical_crossentropy(x, predictions) * temp_mask
        res = tf.keras.backend.sum(prob, axis=-1)
        return res

    def call(self, x):
        z, mean, logvar = self.encoding(x)

        # calculate KL divergence
        kl_loss = self.kld_loss(z, mean, logvar)

        predictions = self.decoder_training(x, z)

        rec_loss = self.reconstruction_loss(x, predictions)
        elbo = kl_loss + rec_loss
        return predictions, kl_loss, rec_loss, elbo, mean, logvar

    def train(self, optimizer, epochs, ckpt_man, ckpt_dir, train_dataset, val_dataset, beta, C_target, C_step=5000):
        @tf.function
        def train_step(x, C_value):
            with tf.GradientTape() as tape:
                kl_loss, rec_loss, elbo = self(x)[1:4]

                loss = tf.keras.backend.mean(tf.keras.backend.abs(kl_loss - C_value) * beta + rec_loss)

            grads = tape.gradient(loss, self.weights)
            optimizer.apply_gradients(zip(grads, self.weights))

            elbo = tf.keras.backend.mean(elbo)
            kl_loss = tf.keras.backend.mean(kl_loss)
            rec_loss = tf.keras.backend.mean(rec_loss)
            return kl_loss, rec_loss, elbo, loss

        @tf.function
        def test_step(x):
            kl_loss, rec_loss, elbo = self(x)[1:4]

            loss = tf.keras.backend.mean(tf.keras.backend.abs(kl_loss - C_target) * beta + rec_loss)
            elbo = tf.keras.backend.mean(elbo)
            kl_loss = tf.keras.backend.mean(kl_loss)
            rec_loss = tf.keras.backend.mean(rec_loss)
            return kl_loss, rec_loss, elbo, loss

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
            total_loss / (step + 1), total_kl_loss / (step + 1), total_rec_loss / (step + 1), total_elbo / (
                        step + 1)
        print("loss:{:.4f} kl_loss:{:.4f} rec_loss:{:.4f} elbo:{:.4f}".format(
            val_loss, val_kl_loss, val_rec_loss, val_elbo))

        if epochs <= 0:
            ckpt_man.save()

        # please refer to https://keras.io/guides/writing_a_training_loop_from_scratch/
        step_count = 1

        print('loss=beta({:f})*|kl-C({:f})|+rec)'.format(beta, C_target))

        for epoch in range(1, epochs+1):
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
                kl_loss, rec_loss, elbo, loss = train_step(x_batch_train, tf.constant(C_value*1.0))

                total_loss = total_loss + loss
                total_elbo = total_elbo + elbo
                total_rec_loss = total_rec_loss + rec_loss
                total_kl_loss = total_kl_loss + kl_loss

                if step_count % 100 == 0:
                    print("step:{:d} train_loss:{:.4f} train_kl_loss:{:.4f} train_rec_loss:{:.4f} train_elbo:{:.4f}"
                          .format(step_count, loss, kl_loss, rec_loss, elbo))

                step_count = step_count + 1

            train_loss, train_kl_loss, train_rec_loss, train_elbo = \
                total_loss/(step+1), total_kl_loss/(step+1), total_rec_loss/(step+1), total_elbo/(step+1)
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
                total_loss/(step+1), total_kl_loss/(step+1), total_rec_loss/(step+1), total_elbo/(step+1)
            with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
                f.write("loss:{:.4f} kl_loss:{:.4f} rec_loss:{:.4f} elbo:{:.4f}\n".format(
                    val_loss, val_kl_loss, val_rec_loss, val_elbo))
            print("loss:{:.4f} kl_loss:{:.4f} rec_loss:{:.4f} elbo:{:.4f}".format(
                val_loss, val_kl_loss, val_rec_loss, val_elbo))

            ckpt_man.save()
            print("time taken:{:.2f}s".format(time.time() - start_time))

        print('training ends, model at {}'.format(ckpt_dir))

    def test(self, test_dataset):
        @tf.function
        def test_step(x):
            kl_loss, rec_loss, elbo = self(x)[1:4]

            elbo = tf.keras.backend.mean(elbo)
            kl_loss = tf.keras.backend.mean(kl_loss)
            rec_loss = tf.keras.backend.mean(rec_loss)
            return kl_loss, rec_loss, elbo

        print("model test")
        total_kl_loss = 0
        total_rec_loss = 0
        total_elbo = 0
        for step, x_batch_test in enumerate(test_dataset):
            kl_loss, rec_loss, elbo = test_step(x_batch_test)

            total_elbo = total_elbo + elbo
            total_rec_loss = total_rec_loss + rec_loss
            total_kl_loss = total_kl_loss + kl_loss

            if step == 0:
                all_mean = self.encoding(x_batch_test)[-2]
            else:
                mean = self.encoding(x_batch_test)[-2]
                all_mean = tf.keras.backend.concatenate([all_mean, mean], axis=0)

        test_kl_loss, test_rec_loss, test_elbo = \
            total_kl_loss / (step + 1), total_rec_loss / (step + 1), total_elbo / (step + 1)
        print("kl_loss:{:.4f} rec_loss:{:.4f} elbo:{:.4f}".format(test_kl_loss, test_rec_loss, test_elbo))

        all_mean = all_mean.numpy()
        cov = np.cov(all_mean, rowvar=False)
        s = []
        n = []
        for i in range(0, cov.shape[0]):
            if cov[i][i] > 0.01:
                s.append(i + 1)
            else:
                n.append(i + 1)
        print('{} active units:{}'.format(len(s), s))
        print('{} inactive units:{}'.format(len(n), n))
        return test_kl_loss, test_rec_loss, test_elbo, len(s)

    def get_mean_representation(self, test_dataset):
        print("get mean vector (representation) for sentences")
        initial = True
        for x_batch_test in test_dataset:
            if initial:
                all_mean = self.encoding(x_batch_test)[-2]
                initial = False
            else:
                mean = self.encoding(x_batch_test)[-2]
                all_mean = tf.keras.backend.concatenate([all_mean, mean], axis=0)
        all_mean = all_mean.numpy()
        return all_mean

    def prior_sampling(self, sample_number):
        if self.prior == 'ig':
            return tf.random.normal(shape=(sample_number, self.z_dim))
        else:
            return self.MoG_prior(sample_number)

    def greedy_decoding(self, z, maxlen):
        # 1 is the index of <bos>
        y = tf.constant(1, shape=(z.shape[0], 1))
        state = None

        res = tf.constant(0, shape=(z.shape[0], 0), dtype=tf.int64)
        for _ in range(0, maxlen):
            dec_embeddings = self.embeddings(y)
            new_z = tf.keras.backend.repeat(z, dec_embeddings.shape[1])
            dec_input = tf.keras.layers.concatenate([dec_embeddings, new_z], axis=-1)
            output, h, c = self.decoder_rnn(dec_input, initial_state=state)
            state = [h, c]
            pred = self.decoder_vocab_prob(output)
            y = tf.keras.backend.argmax(pred, axis=-1)
            res = tf.keras.backend.concatenate([res, y], axis=-1)
        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training script', epilog='start training')
    parser.add_argument('-e', '--emb_dim', default=200, type=int, help='embedding dimensions, default: 200')
    parser.add_argument('-r', '--rnn_dim', default=512, type=int, help='RNN dimensions, default: 512')
    parser.add_argument('-z', '--z_dim', default=32, type=int, help='latent space dimensions, default: 32')
    parser.add_argument('-b', '--batch', default=128, type=int, help='batch size, default: 128')
    parser.add_argument('-lr', '--learning_rate', default=0.0005, type=float, help='learning rate, default: 0.0005')
    parser.add_argument('--epochs', default=20, type=int, help='epochs number, default: 20')
    parser.add_argument('--datapath', default='CBT', help='path of data under dataset directory, default: CBT')
    parser.add_argument('-pr', '--prior', default='ig', help='prior, default: Isotropic Gaussian')
    parser.add_argument('-po', '--posterior', default='diag', help='posterior, default: Diagonal Gaussian')
    parser.add_argument('-beta', default=1, type=float, help='beta for training VAE, default: 1')
    parser.add_argument('-C', default=0, type=float, help='C for training VAE, default: 0')
    parser.add_argument('-s', '--seed', default=0, type=int, help='global random seed')
    parser.add_argument('-m', '--mpath', default='test', help='path of model')

    args = parser.parse_args()

    seed = args.seed
    batch_size = args.batch
    lr = args.learning_rate
    epochs = args.epochs
    datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), args.datapath)
    ckpt_dir = os.path.join(os.path.join(os.getcwd(), 'model'), args.mpath)

    if os.system('mkdir ' + ckpt_dir) != 0:
        print('This is not first training.')
        exit()

    # https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    word2index, index2word, train_dataset, val_dataset, test_dataset = load_data(batch_size, datapath, is_train=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    beta = args.beta
    C = args.C
    emb_dim = args.emb_dim
    rnn_dim = args.rnn_dim
    z_dim = args.z_dim
    prior = args.prior
    posterior = args.posterior
    model = LSTMVAE(emb_dim=emb_dim, rnn_dim=rnn_dim, z_dim=z_dim, vocab_size=len(word2index), prior=prior, post=posterior)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_man = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, max_to_keep=1, checkpoint_name='ckpt')

    with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
        f.write("training configure: prior {}, posterior {}, embedding dimension {:d}, RNN dimension {:d}, "
                "z dimension {:d}, batch size {:d}, epoch number {:d}, learning rate {:f}, "
                "beta {:f}, C {:f}, dataset {}, vocabulary size {:d}\n"
                .format(prior, posterior, emb_dim, rnn_dim, z_dim, batch_size, epochs, lr, beta, C, datapath, len(word2index)))

    model.train(optimizer, epochs, ckpt_man, ckpt_dir, train_dataset, val_dataset, beta, C)

    if not os.path.exists(os.path.join(os.path.join(os.getcwd(), 'model'), 'basic.txt')):
        df = pd.DataFrame(columns=['Model', 'KL', 'Rec.', 'AU', 'PPL'])
        df.to_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'basic.txt'), index=False, float_format='%.6f')
    df = pd.read_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'basic.txt'))

    if os.path.basename(ckpt_dir) in list(df['Model']):
        print('Results already exists.')
        exit()
    else:
        dic = {'Model': os.path.basename(ckpt_dir)}

        kl, rec, elbo, au = model.test(test_dataset)
        dic['KL'] = float(kl)
        dic['Rec.'] = float(rec)
        dic['AU'] = au
        df = pd.concat([df, pd.DataFrame(dic, index=[0])], ignore_index=True)
        df.to_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'basic.txt'), index=False, float_format='%.6f')

    representations = model.get_mean_representation(test_dataset)
    mean_df = pd.DataFrame(representations)
    mean_df.columns = ['dim' + str(i) for i in range(1, z_dim + 1)]
    mean_df.to_csv(os.path.join(ckpt_dir, 'representation.csv'), index_label='index')

