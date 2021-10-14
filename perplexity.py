import os
import random
import argparse
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from utils import load_data


class LSTMLM(tf.keras.Model):
    def __init__(self, emb_dim, rnn_dim, vocab_size):
        super().__init__()
        self.embeddings = tf.keras.layers.Embedding(vocab_size, emb_dim, mask_zero=True)
        self.rnn = tf.keras.layers.LSTM(rnn_dim, return_state=False, return_sequences=True,
                                        kernel_initializer='lecun_normal', recurrent_initializer='lecun_normal')
        self.vocab_prob = tf.keras.layers.Dense(vocab_size, activation='softmax')

    @staticmethod
    def reconstruction_loss(x, predictions):
        # ignore padding
        temp_mask = 1 - tf.keras.backend.cast_to_floatx(tf.keras.backend.equal(x, 0))
        prob = tf.keras.backend.sparse_categorical_crossentropy(x, predictions) * temp_mask
        res = tf.keras.backend.sum(prob, axis=-1)
        return res

    def call(self, x):
        y = tf.keras.backend.concatenate([tf.constant(1, shape=(x.shape[0], 1)), x[:, :-1]], axis=-1)
        embeddings = self.embeddings(y)
        mask = self.embeddings.compute_mask(y)
        rnn_output = self.rnn(embeddings, mask=mask)

        predictions = self.vocab_prob(rnn_output)
        rec_loss = self.reconstruction_loss(x, predictions)
        return rec_loss

    def train(self, optimizer, epochs, train_dataset, val_dataset, ckpt_man=None):
        @tf.function
        def train_step(x):
            with tf.GradientTape() as tape:
                loss = self(x)
                loss = tf.keras.backend.mean(loss)

            grads = tape.gradient(loss, self.weights)
            optimizer.apply_gradients(zip(grads, self.weights))
            return loss

        @tf.function
        def test_step(x):
            loss = self(x)
            return tf.keras.backend.mean(loss)

        total_loss = 0
        for step, x_batch_val in enumerate(val_dataset):
            loss = test_step(x_batch_val)
            total_loss = total_loss + loss

        val_loss = total_loss / (step + 1)
        print("loss:{:.4f}".format(val_loss))

        # please refer to https://keras.io/guides/writing_a_training_loop_from_scratch/
        step_count = 1

        for epoch in range(1, epochs + 1):
            print("Start of epoch {:d}".format(epoch))
            start_time = time.time()

            total_loss = 0
            for step, x_batch_train in enumerate(train_dataset):
                loss = train_step(x_batch_train)

                total_loss = total_loss + loss

                if step_count % 100 == 0:
                    print("step:{:d} train_loss:{:.4f}".format(step_count, loss))

                step_count = step_count + 1

            train_loss = total_loss / (step + 1)
            print("train_loss:{:.4f}".format(train_loss))

            total_loss = 0
            for step, x_batch_val in enumerate(val_dataset):
                loss = test_step(x_batch_val)
                total_loss = total_loss + loss

            val_loss = total_loss / (step + 1)
            print("loss:{:.4f}".format(val_loss))

            if ckpt_man is not None:
                ckpt_man.save()
            print("time taken:{:.2f}s".format(time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='perplexity')
    parser.add_argument('-s', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-m', '--mpath', default='test', help='path of model')

    args = parser.parse_args()

    seed = args.seed
    model_path = args.mpath

    if not os.path.exists(os.path.join(os.path.join(os.getcwd(), 'model'), 'perplexity.txt')):
        df = pd.DataFrame(columns=['Model', 'Forward', 'Reverse'])
        df.to_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'perplexity.txt'), index=False, float_format='%.6f')
    df = pd.read_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'perplexity.txt'))

    if os.path.basename(model_path) in list(df['Model']):
        print('Results already exists.')
        exit()
    else:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        dic = {'Model': os.path.basename(model_path)}
        with open(os.path.join(model_path, 'epoch_loss.txt'), 'r') as f:
            s = f.readlines()[0]

        if 'type' not in s:
            model_type = 'VAE'
        else:
            model_type = s.split(',')[0].split()[-1]

        if model_type == 'VAE':
            s = s.split(',')
            prior = s[0].split()[-1]
            posterior = s[1].split()[-1]
            emb_dim = int(s[2].split()[-1])
            rnn_dim = int(s[3].split()[-1])
            z_dim = int(s[4].split()[-1])
            batch_size = int(s[5].split()[-1])
            lr = float(s[7].split()[-1])
            datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), os.path.basename(s[-2].split()[-1]))
            vocab_size = int(s[-1].split()[-1])
        elif model_type == 'AE':
            s = s.split(',')
            emb_dim = int(s[1].split()[-1])
            rnn_dim = int(s[2].split()[-1])
            z_dim = int(s[3].split()[-1])
            batch_size = int(s[4].split()[-1])
            lr = float(s[6].split()[-1])
            datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), os.path.basename(s[-2].split()[-1]))
            vocab_size = int(s[-1].split()[-1])
        else:
            s = s.split(',')
            emb_dim = int(s[1].split()[-1])
            rnn_dim = int(s[2].split()[-1])
            z_dim = int(s[3].split()[-1])
            batch_size = int(s[4].split()[-1])
            lr = float(s[6].split()[-1])
            datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), os.path.basename(s[-2].split()[-1]))
            vocab_size = int(s[-1].split()[-1])

        word2index, index2word, train_dataset, val_dataset, test_dataset = load_data(batch_size, datapath,
                                                                                     is_train=True)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model = LSTMLM(emb_dim, rnn_dim, vocab_size)

        lm_path = os.path.join(os.path.join(os.getcwd(), 'model'), os.path.basename(datapath) + '_lm')
        if os.path.exists(lm_path):
            ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
            ckpt.restore(tf.train.latest_checkpoint(lm_path)).expect_partial()
        else:
            ckpt_dir = lm_path
            ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
            ckpt_man = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, max_to_keep=1, checkpoint_name='ckpt')

            model.train(optimizer, 10, train_dataset, val_dataset, ckpt_man)

            total_ppl = 0
            count = 0
            for step, x_batch in enumerate(test_dataset):
                sen_len = tf.math.count_nonzero(x_batch, axis=1)
                batch_ppl = tf.keras.backend.exp(model(x_batch) / tf.keras.backend.cast_to_floatx(sen_len))
                count += batch_ppl.shape[0]
                total_ppl = total_ppl + tf.keras.backend.sum(batch_ppl)

            df = pd.concat([df, pd.DataFrame(
                {'Model': os.path.basename(datapath) + 'Real', 'Forward': float(total_ppl / count)}, index=[0])], ignore_index=True)

        # test data
        sentences = []
        maxlen = 0
        with open(os.path.join(model_path, 'generation.txt'), 'r') as f:
            for sentence in f.readlines():
                sentence = sentence.rstrip() + ' <eos>'
                sentence = sentence.split()
                for i in range(len(sentence)):
                    sentence[i] = word2index[sentence[i]]
                if len(sentence) > maxlen:
                    maxlen = len(sentence)
                sentences.append(sentence)

        x_gen = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=maxlen, padding='post',
                                                              truncating='post')

        gen_dataset = tf.data.Dataset.from_tensor_slices(x_gen)
        gen_dataset = gen_dataset.batch(batch_size)

        total_ppl = 0
        count = 0
        for step, x_batch in enumerate(gen_dataset):
            sen_len = tf.math.count_nonzero(x_batch, axis=1)
            batch_ppl = tf.keras.backend.exp(model(x_batch) / tf.keras.backend.cast_to_floatx(sen_len))
            count += batch_ppl.shape[0]
            total_ppl = total_ppl + tf.keras.backend.sum(batch_ppl)

        dic['Forward'] = float(total_ppl / count)

        gen_dataset = gen_dataset.shuffle(len(x_gen))
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model = LSTMLM(emb_dim, rnn_dim, vocab_size)

        model.train(optimizer, 10, gen_dataset, val_dataset)

        total_ppl = 0
        count = 0
        for step, x_batch in enumerate(test_dataset):
            sen_len = tf.math.count_nonzero(x_batch, axis=1)
            batch_ppl = tf.keras.backend.exp(model(x_batch) / tf.keras.backend.cast_to_floatx(sen_len))
            count += batch_ppl.shape[0]
            total_ppl = total_ppl + tf.keras.backend.sum(batch_ppl)

        dic['Reverse'] = float(total_ppl / count)

        print(dic)
        df = pd.concat([df, pd.DataFrame(dic, index=[0])], ignore_index=True)
        df.to_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'perplexity.txt'), index=False, float_format='%.6f')
