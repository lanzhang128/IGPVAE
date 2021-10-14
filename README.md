# LSTM-VAE Implementation and Relevant Evaluations
Before using any file in this repository, please create two directories under the root directory named ''Dataset'' and ''model'', respectively. The Dataset directory is used to storage datasets. The model directory is used to storage models and relevant evaluation results.

## External Package Required
Tensorflow 2, Numpy, Pandas, Scikit-Learn, NLTK, Matplotlib.

## Python File Usage
#### lstm_vae.py
VAE training. Type "python lstm_vae.py -h" for help of training configuration. The dataset path is the relative path under Dataset directory. The trained model path is going to be the relative path under model directory.
#### lstm_ae.py
AE training. Type "python lstm_ae.py -h" for help of training configuration.
#### quality.py
Qualitative evaluation for VAE models including word imputation, homotopy and generation.
#### reconstruction.py
Using mean representation to reconstruct test set and calculate BLEU and Rouge scores.
#### agreement.py
Training a text classifer as well as evaluating on reconstruction.
#### classification.py
Using a 2-hidden-layer MLP with 128 neurons and ReLU activation for classification task.
#### perplexity.py
Calculate forward and reverse perplexity on generated sentences.
#### mnist.py
Train and evaluate on image datasets.
#### ablation.py
Ablation study.
#### aggregated.py
Some estimation on aggregated posterior.
#### robustness.py
Randomly delete 30% of words to evaluate robustness.
#### utils.py
Commonly used functions.

### Example of Usage
This is an example of training and evaluating a VAE trained on a dataset.

First: "python lstm_vae.py -e 200 -r 512 -z 32 -b 128 -lr 0.0005 --epochs 20 --datapath CBT -C 5 -s 0 -po diag -m CBT_C_5_po_diag_0"

This will create a directory named CBT_C_5_po_diag_0 under the model directory. The model will be stored in this directory as well as an epoch_loss.txt file to record losses during training.

Second: "python quality.py -tm 2 -m CBT_C_5_po_diag_0"

This will generate 100K sentences using prior.

Third: "python reconstruction.py -m CBT_C_5_po_diag_0"

This will reconstruct sentences in test set and write them in mean.txt. This will also record BLEU and Rouge scores after reconstruction.