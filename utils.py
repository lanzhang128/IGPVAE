import os
import tensorflow as tf


def load_data(batch_size, path, is_train=False):
    print("loading data")
    word2index = {'<pad>': 0, '<bos>': 1, '<eos>': 2}
    index2word = {0: '<pad>', 1: '<bos>', 2: '<eos>'}
    index = 3
    with open(os.path.join(path, 'vocab.txt'), 'r') as f:
        for vocab in f.readlines():
            vocab = vocab.rstrip()
            word2index[vocab] = index
            index2word[index] = vocab
            index = index + 1

    returns = [word2index, index2word]

    if is_train:
        # training data
        sentences = []
        maxlen = 0
        with open(os.path.join(path, 'train.txt'), 'r') as f:
            for sentence in f.readlines():
                sentence = sentence.rstrip() + ' <eos>'
                sentence = sentence.split()
                for i in range(len(sentence)):
                    sentence[i] = word2index[sentence[i]]
                if len(sentence) > maxlen:
                    maxlen = len(sentence)
                sentences.append(sentence)

        x_train = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=maxlen, padding='post', truncating='post')

        train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
        train_dataset = train_dataset.shuffle(len(x_train)).batch(batch_size)

        # validation data
        sentences = []
        maxlen = 0
        with open(os.path.join(path, 'valid.txt'), 'r') as f:
            for sentence in f.readlines():
                sentence = sentence.rstrip() + ' <eos>'
                sentence = sentence.split()
                for i in range(len(sentence)):
                    sentence[i] = word2index[sentence[i]]
                if len(sentence) > maxlen:
                    maxlen = len(sentence)
                sentences.append(sentence)

        x_val = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=maxlen, padding='post', truncating='post')

        val_dataset = tf.data.Dataset.from_tensor_slices(x_val)
        val_dataset = val_dataset.shuffle(len(x_val)).batch(batch_size)

        returns += [train_dataset, val_dataset]

    # test data
    sentences = []
    maxlen = 0
    with open(os.path.join(path, 'test.txt'), 'r') as f:
        for sentence in f.readlines():
            sentence = sentence.rstrip() + ' <eos>'
            sentence = sentence.split()
            for i in range(len(sentence)):
                sentence[i] = word2index[sentence[i]]
            if len(sentence) > maxlen:
                maxlen = len(sentence)
            sentences.append(sentence)

    x_test = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=maxlen, padding='post', truncating='post')

    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    test_dataset = test_dataset.batch(batch_size)
    returns.append(test_dataset)

    return returns


def load_model(model_path):
    from lstm_vae import LSTMVAE
    from lstm_ae import LSTMAE
    from lstm_iwae import LSTMIWAE

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

        model = LSTMVAE(emb_dim=emb_dim, rnn_dim=rnn_dim, z_dim=z_dim, vocab_size=vocab_size, prior=prior, post=posterior)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
        ckpt.restore(tf.train.latest_checkpoint(model_path)).expect_partial()

    elif model_type == 'AE':
        s = s.split(',')
        emb_dim = int(s[1].split()[-1])
        rnn_dim = int(s[2].split()[-1])
        z_dim = int(s[3].split()[-1])
        batch_size = int(s[4].split()[-1])
        lr = float(s[6].split()[-1])
        datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), os.path.basename(s[-2].split()[-1]))
        vocab_size = int(s[-1].split()[-1])

        model = LSTMAE(emb_dim=emb_dim, rnn_dim=rnn_dim, z_dim=z_dim, vocab_size=vocab_size)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
        ckpt.restore(tf.train.latest_checkpoint(model_path)).expect_partial()
    else:
        s = s.split(',')
        emb_dim = int(s[1].split()[-1])
        rnn_dim = int(s[2].split()[-1])
        z_dim = int(s[3].split()[-1])
        batch_size = int(s[4].split()[-1])
        lr = float(s[6].split()[-1])
        datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), os.path.basename(s[-2].split()[-1]))
        vocab_size = int(s[-1].split()[-1])

        model = LSTMIWAE(emb_dim=emb_dim, rnn_dim=rnn_dim, z_dim=z_dim, vocab_size=vocab_size)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
        ckpt.restore(tf.train.latest_checkpoint(model_path)).expect_partial()

    return model, batch_size, datapath