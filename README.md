# Neural Language Model and Machine Translation

In the following project we have coded up a Neural Language Model and a Machine Translation model.

The models are in the following location and should be downloaded into the `models` directory already. [Download Models](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/ritvik_kalra_research_iiit_ac_in/Es_Pt0tMCGJFkZaZqLe7owwBjfQdKQtsRKuYBty7BBlOUw?e=OQ1EGi)

## Datasets
The following were the datasets used for the tasks:
1. **Europarl Corpus** - It is a monolingual English corpus to be used for training the neural language model for English.
2. **TED Talks Corpus** - A subset of the dataset for the English-French translation task in IWSLT 2016.
3. **News Crawl Corpus** - A subset from the News Crawl 2013 corpus for French language.This is a monolingual French corpus for training the neural language model for French.

## Neural Language Model
The Neural language model uses a LSTM layer at its core. The following is the summary
```
RNN(
  (embedding_layer): Embedding(9107, 128)
  (lstm): LSTM(128, 128, num_layers=3, dropout=0.2)
  (fc): Linear(in_features=128, out_features=9107, bias=True)
)
```

To find the probabilities of some sentence from this language model use the following command
```bash
python language_model.py <MODEL_LOCATION>
```

The perplexities for the train and the test set from the language model have been kept in `./perplexities` directory.

## Machine Translation
The Machine Translation model uses Sequence to Sequence model to translate an English sentence to French. The model summary is as follows:
```
Seq2Seq(
  (encoder): Encoder(
    (dropout): Dropout(p=0.5, inplace=False)
    (embedding_layer): Embedding(13211, 16)
    (lstm): LSTM(16, 128, num_layers=2, dropout=0.5)
  )
  (decoder): Decoder(
    (embedding_layer): Embedding(16269, 16)
    (lstm): LSTM(16, 128, num_layers=2, dropout=0.5)
    (linear): Linear(in_features=128, out_features=16269, bias=True)
    (dropout): Dropout(p=0.5, inplace=False)
  )
)
```

To translate an english sentence to french using the model, use the following command.
```bash
python machine_translation.py <MODEL_LOCATION>
```

The **bleu score** for the Train and Test data are kept in the `./bleu_scores` directory.

## Transfer Learning
The Transfer Learning model takes the models from the Language Model. The encoder model is trained on the english dataset and the decoder model is trained on french dataset. The model summary is a follows:
```
Seq2Seq(
  (encoder): Encoder(
    (dropout): Dropout(p=0.5, inplace=False)
    (embedding_layer): Embedding(12824, 128)
    (lstm): LSTM(128, 128, num_layers=3, dropout=0.5)
    (fc): Linear(in_features=128, out_features=12824, bias=True)
  )
  (decoder): Decoder(
    (embedding_layer): Embedding(15821, 128)
    (lstm): LSTM(128, 128, num_layers=3, dropout=0.5)
    (fc): Linear(in_features=128, out_features=15821, bias=True)
    (dropout): Dropout(p=0.5, inplace=False)
  )
)
```

To use the transfer learning model use the following command
```bash
python transfer_learning.py <MODEL_LOCATION>
```

## Models
The `./models` directory consists of different saved models. You can download the models from the link mentioned above and put them in a directory named `models`.

- `LM1.pth` - is the Language model trained on europarl-corpus
- `MT1.pth` - is the Machine Translation model trained on the ted-talks-corpus
- `MT2.pth` - is the Transfer Learning model