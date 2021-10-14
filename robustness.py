import os
import random
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from utils import load_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='classification task')
    parser.add_argument('-s', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-m', '--mpath', default='test', help='path of model')

    args = parser.parse_args()

    seed = args.seed
    model_path = args.mpath

    if not os.path.exists(os.path.join(os.path.join(os.getcwd(), 'model'), 'robustness.txt')):
        df = pd.DataFrame(columns=['Model', 'dropout_Mean', 'dropout_Std'])
        df.to_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'robustness.txt'), index=False, float_format='%.6f')
    df = pd.read_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'robustness.txt'))

    if os.path.basename(model_path) in list(df['Model']):
        print('Results already exists.')
        exit()
    else:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        dic = {'Model': os.path.basename(model_path)}

        model, batch_size, datapath = load_model(model_path)

        word2index = {'<pad>': 0, '<bos>': 1, '<eos>': 2}
        index2word = {0: '<pad>', 1: '<bos>', 2: '<eos>'}
        index = 3
        with open(os.path.join(datapath, 'vocab.txt'), 'r') as f:
            for vocab in f.readlines():
                vocab = vocab.rstrip()
                word2index[vocab] = index
                index2word[index] = vocab
                index = index + 1

        dropout_sentences = []
        maxlen = 0
        with open(os.path.join(datapath, 'test.txt'), 'r') as f:
            for sentence in f.readlines():
                sentence = sentence.rstrip()
                sentence = sentence.split()
                dropout_sentence = list(
                    np.array(sentence)[sorted(random.sample(range(len(sentence)), int(0.7 * (len(sentence)))))])
                dropout_sentence.append('<eos>')
                for i in range(len(dropout_sentence)):
                    dropout_sentence[i] = word2index[dropout_sentence[i]]
                if len(dropout_sentence) > maxlen:
                    maxlen = len(dropout_sentence)
                dropout_sentences.append(dropout_sentence)

        x_test = tf.keras.preprocessing.sequence.pad_sequences(dropout_sentences, maxlen=maxlen, padding='post',
                                                               truncating='post')

        dropout_dataset = tf.data.Dataset.from_tensor_slices(x_test)
        dropout_dataset = dropout_dataset.batch(batch_size)

        x_dropout = model.get_mean_representation(dropout_dataset)

        labels = []
        with open(os.path.join(datapath, 'test_class_label.txt'), 'r') as f:
            for label in f.readlines():
                label = int(label.rstrip())
                labels.append(label)
        y_gold = np.array(labels)
        y_gold = tf.keras.utils.to_categorical(y_gold)

        x, y = shuffle(x_dropout, y_gold, random_state=seed)
        x_train, x_test = x[:int(0.8 * x.shape[0]), :], x[int(0.8 * x.shape[0]):, :]
        y_train, y_test = y[:int(0.8 * y.shape[0]), :], y[int(0.8 * y.shape[0]):, :]
        print("training points: {:d}, test points: {:d}".format(x_train.shape[0], x_test.shape[0]))
        acc = []
        for i in range(0, 10):
            inputs = tf.keras.Input(shape=(x.shape[1],))
            d1 = tf.keras.layers.Dense(128, activation='relu')(inputs)
            d2 = tf.keras.layers.Dense(128, activation='relu')(d1)
            outputs = tf.keras.layers.Dense(y.shape[1], activation='softmax')(d2)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                          loss="categorical_crossentropy", metrics=['accuracy'])
            model.fit(x_train, y_train, batch_size=64, epochs=20, validation_split=0.2, verbose=0)
            test_scores = model.evaluate(x_test, y_test, verbose=0)
            acc.append(test_scores[1])
        acc = np.array(acc)
        dic['dropout_Mean'] = np.mean(acc)
        dic['dropout_Std'] = np.std(acc)
        print(dic)
        df = pd.concat([df, pd.DataFrame(dic, index=[0])], ignore_index=True)
        df.to_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'robustness.txt'), index=False, float_format='%.6f')