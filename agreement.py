import os
import random
import argparse
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


class LSTMClassifier(tf.keras.Model):
    def __init__(self, emb_dim, rnn_dim, vocab_size, num_labels):
        super().__init__()
        self.embeddings = tf.keras.layers.Embedding(vocab_size, emb_dim, mask_zero=True)
        self.rnn = tf.keras.layers.LSTM(rnn_dim, return_state=False, return_sequences=True,
                                        kernel_initializer='lecun_normal', recurrent_initializer='lecun_normal')
        self.hidden_layer_1 = tf.keras.layers.Dense(128, activation='relu')
        self.hidden_layer_2 = tf.keras.layers.Dense(128, activation='relu')
        self.clf_layer = tf.keras.layers.Dense(num_labels, activation='softmax')
        self.loss_layer = tf.keras.losses.CategoricalCrossentropy()

    def call(self, x, y=None):
        enc_embeddings = self.embeddings(x)
        mask = self.embeddings.compute_mask(x)
        rnn_output = self.rnn(enc_embeddings, mask=mask)

        # extract the whole sentence representation, 2 is the index of <eos>
        temp_mask = tf.keras.backend.cast_to_floatx(tf.keras.backend.equal(x, 2))
        temp_mask = tf.keras.backend.expand_dims(temp_mask)
        temp_mask = tf.keras.backend.repeat_elements(temp_mask, rnn_output.shape[2], axis=2)
        final_output = tf.keras.backend.sum(rnn_output * temp_mask, axis=1)
        y_pred = self.clf_layer(self.hidden_layer_2(self.hidden_layer_1(final_output)))
        if y is not None:
            loss = tf.keras.losses.categorical_crossentropy(y, y_pred)
            return loss
        else:
            return y_pred

    def train(self, optimizer, epochs, ckpt_man, ckpt_dir, train_dataset, val_dataset):
        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                loss = self(x, y)
                loss = tf.keras.backend.mean(loss)

            grads = tape.gradient(loss, self.weights)
            optimizer.apply_gradients(zip(grads, self.weights))
            return loss

        @tf.function
        def test_step(x, y):
            loss = self(x, y)
            return tf.keras.backend.mean(loss)

        total_loss = 0
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            loss = test_step(x_batch_val, y_batch_val)
            total_loss = total_loss + loss

        val_loss = total_loss / (step + 1)
        print("loss:{:.4f}".format(val_loss))

        # please refer to https://keras.io/guides/writing_a_training_loop_from_scratch/
        step_count = 1

        for epoch in range(1, epochs + 1):
            print("Start of epoch {:d}".format(epoch))
            start_time = time.time()

            total_loss = 0
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                loss = train_step(x_batch_train, y_batch_train)

                total_loss = total_loss + loss

                if step_count % 100 == 0:
                    print("step:{:d} train_loss:{:.4f}".format(step_count, loss))

                step_count = step_count + 1

            train_loss = total_loss / (step + 1)
            print("train_loss:{:.4f}".format(train_loss))

            total_loss = 0
            for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
                loss = test_step(x_batch_val, y_batch_val)
                total_loss = total_loss + loss

            val_loss = total_loss / (step + 1)
            print("loss:{:.4f}".format(val_loss))

            ckpt_man.save()
            print("time taken:{:.2f}s".format(time.time() - start_time))

        print('training ends, model at {}'.format(ckpt_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Agreement')
    parser.add_argument('-s', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-m', '--mpath', default='test', help='path of model')

    args = parser.parse_args()

    seed = args.seed
    model_path = args.mpath

    if not os.path.exists(os.path.join(os.path.join(os.getcwd(), 'model'), 'agreement.txt')):
        df = pd.DataFrame(columns=['Model', 'precision', 'recall', 'F1'])
        df.to_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'agreement.txt'), index=False, float_format='%.6f')
    df = pd.read_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'agreement.txt'))

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

        word2index = {'<pad>': 0, '<bos>': 1, '<eos>': 2}
        index = 3
        with open(os.path.join(datapath, 'vocab.txt'), 'r') as f:
            for vocab in f.readlines():
                vocab = vocab.rstrip()
                word2index[vocab] = index
                index = index + 1

        labels = []
        with open(os.path.join(datapath, 'test_class_label.txt'), 'r') as f:
            for label in f.readlines():
                label = int(label.rstrip())
                labels.append(label)
        y_test = np.array(labels)
        y_test = tf.keras.utils.to_categorical(y_test)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model = LSTMClassifier(emb_dim, rnn_dim, vocab_size, y_test.shape[1])

        clf_path = os.path.join(os.path.join(os.getcwd(), 'model'), os.path.basename(datapath) + '_clf')
        if os.path.exists(clf_path):
            ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
            ckpt.restore(tf.train.latest_checkpoint(clf_path)).expect_partial()
        else:
            ckpt_dir = clf_path
            ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
            ckpt_man = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, max_to_keep=1, checkpoint_name='ckpt')

            # training data
            sentences = []
            maxlen = 0
            with open(os.path.join(datapath, 'train.txt'), 'r') as f:
                for sentence in f.readlines():
                    sentence = '<bos> ' + sentence.rstrip() + ' <eos>'
                    sentence = sentence.split()
                    for i in range(len(sentence)):
                        sentence[i] = word2index[sentence[i]]
                    if len(sentence) > maxlen:
                        maxlen = len(sentence)
                    sentences.append(sentence)

            x_train = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=maxlen, padding='post',
                                                                    truncating='post')

            labels = []
            with open(os.path.join(datapath, 'train_class_label.txt'), 'r') as f:
                for label in f.readlines():
                    label = int(label.rstrip())
                    labels.append(label)
            y_train = np.array(labels)
            y_train = tf.keras.utils.to_categorical(y_train)

            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            train_dataset = train_dataset.shuffle(len(x_train)).batch(batch_size)

            # validation data
            sentences = []
            maxlen = 0
            with open(os.path.join(datapath, 'valid.txt'), 'r') as f:
                for sentence in f.readlines():
                    sentence = '<bos> ' + sentence.rstrip() + ' <eos>'
                    sentence = sentence.split()
                    for i in range(len(sentence)):
                        sentence[i] = word2index[sentence[i]]
                    if len(sentence) > maxlen:
                        maxlen = len(sentence)
                    sentences.append(sentence)

            x_valid = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=maxlen, padding='post',
                                                                    truncating='post')

            labels = []
            with open(os.path.join(datapath, 'valid_class_label.txt'), 'r') as f:
                for label in f.readlines():
                    label = int(label.rstrip())
                    labels.append(label)
            y_valid = np.array(labels)
            y_valid = tf.keras.utils.to_categorical(y_valid)

            val_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
            val_dataset = val_dataset.shuffle(len(x_valid)).batch(batch_size)

            model.train(optimizer, 20, ckpt_man, ckpt_dir, train_dataset, val_dataset)

        # test data
        sentences = []
        maxlen = 0
        with open(os.path.join(model_path, 'mean.txt'), 'r') as f:
            for sentence in f.readlines():
                sentence = '<bos> ' + sentence.rstrip() + ' <eos>'
                sentence = sentence.split()
                for i in range(len(sentence)):
                    sentence[i] = word2index[sentence[i]]
                if len(sentence) > maxlen:
                    maxlen = len(sentence)
                sentences.append(sentence)

        x_test = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=maxlen, padding='post',
                                                               truncating='post')

        test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
        test_dataset = test_dataset.batch(batch_size)

        y_pred = tf.zeros(shape=(0, y_test.shape[1]))
        for step, x_batch_test in enumerate(test_dataset):
            y_pred = tf.keras.backend.concatenate([y_pred, model(x_batch_test)], axis=0)

        y_test = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        dic['precision'] = precision_score(y_test, y_pred, average='macro')
        dic['recall'] = recall_score(y_test, y_pred, average='macro')
        dic['F1'] = f1_score(y_test, y_pred, average='macro')
        print(dic)
        df = pd.concat([df, pd.DataFrame(dic, index=[0])], ignore_index=True)
        df.to_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'agreement.txt'), index=False, float_format='%.6f')
