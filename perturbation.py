import os
import random
import argparse
import tensorflow as tf
import numpy as np
import modeling


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model evaluation')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed')
    parser.add_argument('-m', '--mpath', default='snli_capacity_diag', help='path of model')

    args = parser.parse_args()
    seed = args.seed
    model_path = os.path.join(os.path.join(os.getcwd(), 'model'), args.mpath)
    print(model_path)

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open(os.path.join(model_path, 'epoch_loss.txt'), 'r') as f:
        s = f.readlines()[0]
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

    model = modeling.LSTMVAE(emb_dim=emb_dim, rnn_dim=rnn_dim, z_dim=z_dim, vocab_size=vocab_size, prior=prior,
                             post=posterior)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)

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
            sentence = sentence.rstrip() + ' <eos>'
            sentence = sentence.split()
            for i in range(len(sentence)):
                sentence[i] = word2index[sentence[i]]
            if len(sentence) > maxlen:
                maxlen = len(sentence)
            sentences.append(sentence)

    sentences = random.sample(sentences, 1)
    for epoch in range(1, 21):
        ckpt.restore(os.path.join(model_path, 'ckpt-' + str(epoch))).expect_partial()
        f = open(os.path.join(model_path, f'perturbation-{epoch}.txt'), 'w')
        f.write('sentence: ')
        f.write(' '.join([index2word[sentences[0][i]] for i in range(0, len(sentences[0])-1)]) + '\n')

        z = model.encoding(tf.constant(sentences[0], shape=(1, len(sentences[0]))))[1]
        f.write('z: ')
        f.write(' '.join(['%.3f' % z.numpy().tolist()[0][i] for i in range(0, len(z.numpy().tolist()[0]))]) + '\n')

        for dim in range(0, z_dim):
            f.write('dim {:d} homotopy '.format(dim + 1))
            f.write('(from {:.3f} to {:.3f})'.format(z.numpy()[0, dim] - 1, z.numpy()[0, dim] + 1) + '\n')
            for i in range(0, 5):
                f.write(str(i + 1)+'. ')
                temp_z = z.numpy()
                temp_z[0, dim] = (1 - 0.25 * i) * (z.numpy()[0, dim] - 1) + 0.25 * i * (z.numpy()[0, dim] + 1)
                temp_z = tf.constant(temp_z)
                res = model.greedy_decoding(temp_z, maxlen).numpy().tolist()[0]
                for j in range(0, len(res)):
                    if index2word[int(res[j])] == '<eos>':
                        break
                    else:
                        f.write(index2word[int(res[j])] + ' ')
                f.write('\n')

        f.close()