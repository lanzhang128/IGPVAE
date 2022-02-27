import tensorflow as tf
import numpy as np
import random
import argparse
import time
import os
import pandas as pd
from utils import load_data


class LSTMIWAE(tf.keras.Model):
    def __init__(self, emb_dim, rnn_dim, z_dim, vocab_size):
        super().__init__()
        self.embeddings = tf.keras.layers.Embedding(vocab_size, emb_dim, mask_zero=True)
        self.encoder_rnn = tf.keras.layers.LSTM(rnn_dim, return_state=False, return_sequences=True,
                                                kernel_initializer='lecun_normal', recurrent_initializer='lecun_normal')
        self.decoder_rnn = tf.keras.layers.LSTM(rnn_dim, return_state=True, return_sequences=True,
                                                kernel_initializer='lecun_normal', recurrent_initializer='lecun_normal')
        self.decoder_vocab_prob = tf.keras.layers.Dense(vocab_size, activation='softmax')
        self.encoder_mean_layer = tf.keras.layers.Dense(z_dim)
        self.encoder_logvar_layer = tf.keras.layers.Dense(z_dim)

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
        logvar = self.encoder_logvar_layer(output)
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
        post_kld = tf.keras.backend.sum(-0.5 * (logvar + tf.keras.backend.square(z - mean) / tf.keras.backend.exp(logvar)), axis=-1)
        prior_kld = tf.keras.backend.sum(-0.5 * tf.keras.backend.square(z), axis=-1)
        kld = post_kld - prior_kld
        return kld

    @staticmethod
    def reconstruction_loss(x, predictions):
        # ignore padding
        temp_mask = 1 - tf.keras.backend.cast_to_floatx(tf.keras.backend.equal(x, 0))
        prob = tf.keras.backend.sparse_categorical_crossentropy(x, predictions) * temp_mask
        res = tf.keras.backend.sum(prob, axis=-1)
        return res

    def call(self, x, k):
        for i in tf.range(k):
            z, mean, logvar = self.encoding(x)

            # calculate KL divergence
            kl_loss = self.kld_loss(z, mean, logvar)

            predictions = self.decoder_training(x, z)

            rec_loss = self.reconstruction_loss(x, predictions)
            if i == 0:
                elbo = tf.keras.backend.expand_dims(kl_loss + rec_loss)
            else:
                elbo = tf.keras.backend.concatenate([elbo, tf.keras.backend.expand_dims(kl_loss + rec_loss)], axis=-1)

        weights = tf.stop_gradient(tf.keras.backend.softmax(- elbo, axis=-1))

        return tf.keras.backend.sum(weights * elbo, axis=-1)

    def train(self, optimizer, epochs, ckpt_man, ckpt_dir, train_dataset, val_dataset, k):
        total_loss = 0
        for step, x_batch_val in enumerate(val_dataset):
            loss = self(x_batch_val, k)
            loss = tf.keras.backend.mean(loss)
            total_loss = total_loss + loss

        val_loss = total_loss / (step + 1)
        print("loss:{:.4f}".format(val_loss))

        if epochs <= 0:
            ckpt_man.save()

        # please refer to https://keras.io/guides/writing_a_training_loop_from_scratch/
        step_count = 1

        for epoch in range(1, epochs+1):
            print("Start of epoch {:d}".format(epoch))
            start_time = time.time()

            total_loss = 0
            for step, x_batch_train in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    loss = self(x_batch_train, k)
                    loss = tf.keras.backend.mean(loss)

                grads = tape.gradient(loss, self.weights)
                optimizer.apply_gradients(zip(grads, self.weights))

                total_loss = total_loss + loss

                if step_count % 100 == 0:
                    print("step:{:d} train_loss:{:.4f}".format(step_count, loss))

                step_count = step_count + 1

            train_loss = total_loss/(step+1)
            with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
                f.write("train_loss:{:.4f} ".format(train_loss))
            print("train_loss:{:.4f}".format(train_loss))

            total_loss = 0
            for step, x_batch_val in enumerate(val_dataset):
                loss = self(x_batch_val, k)
                loss = tf.keras.backend.mean(loss)
                total_loss = total_loss + loss

            val_loss = total_loss/(step+1)
            with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
                f.write("loss:{:.4f}\n".format(val_loss))
            print("loss:{:.4f}".format(val_loss))

            ckpt_man.save()
            print("time taken:{:.2f}s".format(time.time() - start_time))

        print('training ends, model at {}'.format(ckpt_dir))

    def test(self, test_dataset):
        @tf.function
        def test_step(x):
            z, mean, logvar = self.encoding(x)

            # calculate KL divergence
            kl_loss = self.kld_loss(z, mean, logvar)
            predictions = self.decoder_training(x, z)
            rec_loss = self.reconstruction_loss(x, predictions)
            elbo = kl_loss + rec_loss

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
    parser.add_argument('-k', default=1, type=int, help='k for training IWAE, default: 1')
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
    k = args.k
    emb_dim = args.emb_dim
    rnn_dim = args.rnn_dim
    z_dim = args.z_dim
    model = LSTMIWAE(emb_dim=emb_dim, rnn_dim=rnn_dim, z_dim=z_dim, vocab_size=len(word2index))
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_man = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, max_to_keep=1, checkpoint_name='ckpt')

    with open(os.path.join(ckpt_dir, 'epoch_loss.txt'), 'a') as f:
        f.write("training configure: type IWAE, embedding dimension {:d}, RNN dimension {:d}, "
                "z dimension {:d}, batch size {:d}, epoch number {:d}, learning rate {:f}, "
                "k {:d}, dataset {}, vocabulary size {:d}\n"
                .format(emb_dim, rnn_dim, z_dim, batch_size, epochs, lr, k, datapath, len(word2index)))

    model.train(optimizer, epochs, ckpt_man, ckpt_dir, train_dataset, val_dataset, k)

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

