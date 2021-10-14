import os
import argparse
import pandas as pd
from utils import load_data, load_model
from nltk.translate.bleu_score import corpus_bleu
from rouge import rouge as corpus_rouge


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model evaluation')
    parser.add_argument('-m', '--mpath', default='test', help='path of model')

    args = parser.parse_args()
    model_path = args.mpath

    if not os.path.exists(os.path.join(os.path.join(os.getcwd(), 'model'), 'bleu_rouge.txt')):
        df = pd.DataFrame(columns=['Model', 'BLEU-1', 'BLEU-2', 'BLEU-4'])
        df.to_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'bleu_rouge.txt'), index=False, float_format='%.6f')
    df = pd.read_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'bleu_rouge.txt'))

    if os.path.basename(model_path) in list(df['Model']):
        print('Results already exists.')
        exit()
    else:
        dic = {'Model': os.path.basename(model_path)}

        reconstruction_file = os.path.join(model_path, 'mean.txt')
        model, batch_size, datapath = load_model(model_path)
        word2index, index2word, test_dataset = load_data(batch_size, datapath, is_train=False)

        f = open(reconstruction_file, 'w')
        for x_batch_test in test_dataset:
            mean = model.encoding(x_batch_test)[-2]
            res = model.greedy_decoding(mean, x_batch_test.shape[1])
            res = res.numpy().tolist()
            for element in res:
                if 2 in element:
                    element = element[:element.index(2)]
                element = [index2word[i] for i in element]
                f.write(' '.join(element) + '\n')
        f.close()

        bleu_references = []
        rouge_references = []
        with open(os.path.join(datapath, 'test.txt'), 'r') as f:
            for sentence in f.readlines():
                rouge_references.append(sentence.rstrip())
                bleu_references.append([sentence.rstrip().split()])

        bleu_candidates = []
        rouge_candidates = []
        with open(reconstruction_file, 'r') as f:
            for sentence in f.readlines():
                rouge_candidates.append(sentence.rstrip())
                bleu_candidates.append(sentence.rstrip().split())

        dic['BLEU-1'] = corpus_bleu(bleu_references, bleu_candidates, weights=(1, 0, 0, 0))
        dic['BLEU-2'] = corpus_bleu(bleu_references, bleu_candidates, weights=(0.5, 0.5, 0, 0))
        dic['BLEU-4'] = corpus_bleu(bleu_references, bleu_candidates, weights=(0.25, 0.25, 0.25, 0.25))
        rouge_score = corpus_rouge(rouge_candidates, rouge_references)
        dic['ROUGE-1-F1'] = rouge_score['rouge_1/f_score']
        dic['ROUGE-2-F1'] = rouge_score['rouge_2/f_score']
        dic['ROUGE-4-F1'] = rouge_score['rouge_4/f_score']
        print(dic)
        df = pd.concat([df, pd.DataFrame(dic, index=[0])], ignore_index=True)
        df.to_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'bleu_rouge.txt'), index=False, float_format='%.6f')