import os
import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='classification task')
    parser.add_argument('-m', '--mpath', default='test', help='path of model')

    args = parser.parse_args()
    model_path = os.path.join(os.path.join(os.getcwd(), 'model'), args.mpath)

    if not os.path.exists(os.path.join(os.path.join(os.getcwd(), 'model'), 'iphr.txt')):
        df = pd.DataFrame(columns=['Model'])
        df.to_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'iphr.txt'), index=False,
                  float_format='%.6f')
    df = pd.read_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'iphr.txt'))

    if os.path.basename(model_path) in list(df['Model']):
        print('Results already exists.')
        exit()
    else:
        dic = {'Model': os.path.basename(model_path)}
        with open(os.path.join(model_path, 'epoch_loss.txt'), 'r') as f:
            s = f.readlines()[0]
            s = s.split(',')

        datapath = os.path.join(os.path.join(os.getcwd(), 'Dataset'), os.path.basename(s[-2].split()[-1]))
        reference_path = os.path.join(datapath, 'test.txt')
        references = []
        with open(reference_path, 'r') as f:
            for sentence in f.readlines():
                sentence = sentence.rstrip().split()
                references.append(sentence)

        length = max([len(references[i]) for i in range(0, len(references))])

        candidate_path = os.path.join(model_path, 'mean.txt')
        candidates = []
        with open(candidate_path, 'r') as f:
            for sentence in f.readlines():
                sentence = sentence.rstrip().split()
                candidates.append(sentence)

        count_match = [0 for _ in range(0, length)]
        count = [0 for _ in range(0, length)]
        for reference, candidate in zip(references, candidates):
            temp = [int(reference[i] == candidate[i]) for i in range(min(len(reference), len(candidate)))]
            for i in range(min(len(count), len(temp))):
                count_match[i] += temp[i]
            for i in range(len(reference)):
                count[i] += 1

        for i in range(0, len(count)):
            count_match[i] = count_match[i] / count[i]
            dic['pos'+str(i+1)] = count_match[i]

        df = pd.concat([df, pd.DataFrame(dic, index=[0])], ignore_index=True)
        df.to_csv(os.path.join(os.path.join(os.getcwd(), 'model'), 'iphr.txt'), index=False, float_format='%.6f')