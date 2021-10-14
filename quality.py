import os
import random
import argparse
import tensorflow as tf
from utils import load_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model evaluation')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed')
    parser.add_argument('-tm', '--test_mode', type=int, default=0, help='test mode: 0 for imputing missing words,'
                                                                        '1 for normal and dimension-wise homotopy,'
                                                                        '2 for random generation.')
    parser.add_argument('-m', '--model_path', default='test', help='path of model')

    args = parser.parse_args()
    seed = args.seed
    test_mode = args.test_mode
    model_path = args.model_path
    print(model_path)

    random.seed(seed)

    model, _, datapath = load_model(model_path)

    word2index = {'<pad>': 0, '<bos>': 1, '<eos>': 2}
    index2word = {0: '<pad>', 1: '<bos>', 2: '<eos>'}
    index = 3
    with open(os.path.join(datapath, 'vocab.txt'), 'r') as f:
        for vocab in f.readlines():
            vocab = vocab.rstrip()
            word2index[vocab] = index
            index2word[index] = vocab
            index = index + 1

    maxlen = 0
    sentences = []
    with open(os.path.join(datapath, 'test.txt'), 'r') as f:
        for sentence in f.readlines():
            if '<unk>' not in sentence:
                sentence = sentence.rstrip() + ' <eos>'
                sentence = sentence.split()
                for i in range(len(sentence)):
                    sentence[i] = word2index[sentence[i]]
                if len(sentence) > maxlen:
                    maxlen = len(sentence)
                sentences.append(sentence)

    if test_mode == 0:
        f = open(os.path.join(model_path, 'imputation.txt'), 'w')
        sentences = random.sample(sentences, 5)
        for k in range(len(sentences)):
            for prop in range(1, 5):
                sentence = sentences[k]
                f.write('origin sentence: \n')
                f.write(' '.join([index2word[sentence[i]] for i in range(0, len(sentence) - 1)]) + '\n')

                sentence = sentence[:int(0.25 * prop * (len(sentence) - 1))]

                f.write('imputed sentence: \n')
                f.write(' '.join([index2word[sentence[i]] for i in range(0, len(sentence))]) + '\n')

                z = model.encoding(tf.constant(sentence, shape=(1, len(sentence))))[1]
                y = tf.constant([1] + sentence, shape=(z.shape[0], len(sentence) + 1), dtype=tf.int64)
                state = None

                res = y[:, 1:]
                for _ in range(maxlen - len(sentence)):
                    dec_embeddings = model.embeddings(y)
                    new_z = tf.keras.backend.repeat(z, dec_embeddings.shape[1])
                    dec_input = tf.keras.layers.concatenate([dec_embeddings, new_z], axis=-1)
                    output, h, c = model.decoder_rnn(dec_input, initial_state=state)
                    state = [h, c]
                    output = output[:, -1:, :]
                    pred = model.decoder_vocab_prob(output)
                    y = tf.keras.backend.argmax(pred, axis=-1)
                    res = tf.keras.backend.concatenate([res, y], axis=-1)

                f.write('completed sentence: \n')
                res = res.numpy().tolist()[0]
                for j in range(0, len(res)):
                    if index2word[int(res[j])] == '<eos>':
                        break
                    else:
                        f.write(index2word[int(res[j])] + ' ')
                f.write('\n')
            f.write('\n')
        f.close()

    elif test_mode == 1:
        sentences = random.sample(sentences, 2)
        f = open(os.path.join(model_path, 'homotopy.txt'), 'w')
        f.write('start sentence: ')
        f.write(' '.join([index2word[sentences[0][i]] for i in range(0, len(sentences[0])-1)]) + '\n')
        f.write('end sentence: ')
        f.write(' '.join([index2word[sentences[1][i]] for i in range(0, len(sentences[1])-1)]) + '\n')

        z1 = model.encoding(tf.constant(sentences[0], shape=(1, len(sentences[0]))))[1]
        z2 = model.encoding(tf.constant(sentences[1], shape=(1, len(sentences[1]))))[1]
        f.write('z1: ')
        f.write(' '.join(['%.3f' % z1.numpy().tolist()[0][i] for i in range(0, len(z1.numpy().tolist()[0]))]) + '\n')
        f.write('z2: ')
        f.write(' '.join(['%.3f' % z2.numpy().tolist()[0][i] for i in range(0, len(z2.numpy().tolist()[0]))]) + '\n')

        f.write('normal homotopy:\n')
        for i in range(0, 6):
            f.write(str(i + 1)+'. ')
            z = (1 - 0.2 * i) * z1 + 0.2 * i * z2
            res = model.greedy_decoding(z, maxlen).numpy().tolist()[0]
            for j in range(0, len(res)):
                if index2word[int(res[j])] == '<eos>':
                    break
                else:
                    f.write(index2word[int(res[j])] + ' ')
            f.write('\n')

        for dim in range(0, z1.shape[1]):
            f.write('dim {:d} homotopy '.format(dim + 1))
            f.write('(from {:.3f} to {:.3f})'.format(z1.numpy()[0, dim], z2.numpy()[0, dim]) + '\n')
            for i in range(0, 5):
                f.write(str(i + 1)+'. ')
                z = z1.numpy()
                z[0, dim] = (1 - 0.25 * i) * z1.numpy()[0, dim] + 0.25 * i * z2.numpy()[0, dim]
                z = tf.constant(z)
                res = model.greedy_decoding(z, maxlen).numpy().tolist()[0]
                for j in range(0, len(res)):
                    if index2word[int(res[j])] == '<eos>':
                        break
                    else:
                        f.write(index2word[int(res[j])] + ' ')
                f.write('\n')
            z1 = z
        f.close()

    elif test_mode == 2:
        f = open(os.path.join(model_path, 'generation.txt'), 'w')
        sample_shape = model.encoding(tf.constant(sentences[0], shape=(1, len(sentences[0]))))[1].shape
        for _ in range(500):
            z = tf.random.normal(shape=(200, sample_shape[1]))
            res = model.greedy_decoding(z, maxlen)
            res = res.numpy().tolist()
            for element in res:
                if 2 in element:
                    element = element[:element.index(2)]
                element = [index2word[i] for i in element]
                f.write(' '.join(element) + '\n')
        f.close()

    else:
        print('Wrong test mode!')
