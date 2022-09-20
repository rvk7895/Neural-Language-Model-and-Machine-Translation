import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy
import spacy
import numpy as np
import random
import math
import time
from tqdm import tqdm
import re
import sys

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

BATCH_SIZE = 16
MAX_EPOCHS = 10
EMBEDDING_SIZE = 128
HIDDEN_DIMENSION = 128
LAYERS = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, en_file_location, fr_file_location, sequence_length):
        self.en_file_location = en_file_location
        self.fr_file_location = fr_file_location
        self.sequence_length = sequence_length
        self.initialize_data()
        self.modify()
        self.combined_data = list()
        self.combineData()

    def initialize_data(self):
        with open(self.en_file_location, "r") as inFile:
            enData = inFile.readlines()
        
        with open(self.fr_file_location, "r") as inFile:
            frData = inFile.readlines()

        en_tokenized_data = self.tokenizer(enData)
        self.ENdataset = self.cleaner(en_tokenized_data)
        (
            self.ENword2Index,
            self.ENindex2Word,
            self.ENvocab_size,
            self.ENvocab,
            self.ENwordFrequency
        ) = self.vocabBuilder(self.ENdataset)
        self.ENwords = list()
        for sentence in self.ENdataset:
            for word in sentence:
                self.ENwords.append(word)

        self.ENwords_indexes = [self.ENword2Index[word] for word in self.ENwords]

        fr_tokenized_data = self.tokenizer(frData)
        self.FRdataset = self.cleaner(fr_tokenized_data)
        (
            self.FRword2Index,
            self.FRindex2Word,
            self.FRvocab_size,
            self.FRvocab,
            self.FRwordFrequency
        ) = self.vocabBuilder(self.FRdataset)
        self.FRwords = list()
        for sentence in self.FRdataset:
            for word in sentence:
                self.FRwords.append(word)

        self.FRwords_indexes = [self.FRword2Index[word] for word in self.FRwords]

    def tokenizer(self,corpus):
        """
            tokenizes the corpus
            
            Arguments:
                corpus (list)

            Returns:
                tokenized corpus (list)
        """
        hashtag_regex = "#[a-zA-Z0-9]+"
        url_regex = "((http|https)://)(www.)?[a-zA-Z0-9@:%._\\+~#?&//=]{2,256}\\.[a-z]{2,6}\\b([-a-zA-Z0-9@:%._\\+~#?&//=]*)"
        mention_regex = "@\w+"

        processed_corpus = list()

        for tweet in corpus:
            normalized_tweet = tweet.lower()
            hashtag_removed_tweet = re.sub(hashtag_regex, "<HASHTAG>", normalized_tweet)
            website_removed_tweet = re.sub(url_regex, "<URL>", hashtag_removed_tweet)
            mention_removed_tweet = re.sub(
                mention_regex, "<MENTION>", website_removed_tweet
            )
            punctuation_repeat_removed = re.sub(
                r"(\W)(?=\1)", "", mention_removed_tweet
            )
            tokenized_tweet = punctuation_repeat_removed.split()

            cleaned_tokenized_tweet = list()
            for token in tokenized_tweet:
                if token not in ["<HASHTAG>", "<URL>", "<MENTION>", "<OOV>"]:
                    split_tokens = "".join(
                        (char if char.isalpha() or char.isnumeric() else f" {char} ")
                        for char in token
                    ).split()
                    for cleaned_token in split_tokens:
                        cleaned_tokenized_tweet.append(cleaned_token)

                else:
                    cleaned_tokenized_tweet.append(token)
            cleaned_tokenized_tweet = ['<SOS>'] + cleaned_tokenized_tweet + ['<EOS>']
            processed_corpus.append(cleaned_tokenized_tweet)

        return processed_corpus

    def cleaner(self,corpus):
        """
            replacing !,?,. with . and removing other punctuations
            
            Arguments:
                tokenized corpuse (list)

            Returns:
                cleaned corpus (list)
        """
        import string

        cleaned_corpus = list()

        for sentence in corpus:
            new_sentence = list()
            for token in sentence:
                if token in ["!", ".", "?"]:
                    new_sentence.append(".")
                elif token in string.punctuation:
                    continue
                else:
                    new_sentence.append(token)

            cleaned_corpus.append(new_sentence)

        return cleaned_corpus

    def vocabBuilder(self,corpus):
        """
            Builds the vocabulary of the input dataset.

            Arguments:
                The cleaned tokenized the dataset
            
            Returns:
                Word to Index dict, Index to Word list, Number of Unique Words, Set of Vocab
        """
        word2Index = dict()
        index2Word = list()
        vocab = set()
        wordFrequency = dict()

        n_unique_words = 0

        for sentence in corpus:
            for word in sentence:
                vocab.add(word)
                if word not in word2Index:
                    word2Index[word] = n_unique_words
                    index2Word.append(word)
                    n_unique_words += 1
                    wordFrequency[word] = 1
                else:
                    wordFrequency[word] += 1

        return word2Index, index2Word, n_unique_words, vocab, wordFrequency
    
    def modify(self):
        for i in range(len(self.ENdataset)):
            for j in range(len(self.ENdataset[i])):
                if self.ENwordFrequency[self.ENdataset[i][j]] < 2:
                    self.ENdataset[i][j] = '<OOV>'
                elif any(character.isdigit() for character in self.ENdataset[i][j]):
                    self.ENdataset[i][j] = '<OOV>'

        print(self.ENvocab_size)
        
        self.ENdataset = self.cleaner(self.ENdataset)
        (
            self.ENword2Index,
            self.ENindex2Word,
            self.ENvocab_size,
            self.ENvocab,
            self.ENwordFrequency
        ) = self.vocabBuilder(self.ENdataset)
        self.ENwords = list()
        for sentence in self.ENdataset:
            for word in sentence:
                self.ENwords.append(word)

        self.ENwords_indexes = [self.ENword2Index[word] for word in self.ENwords]

        for i in range(len(self.FRdataset)):
            for j in range(len(self.FRdataset[i])):
                if self.FRwordFrequency[self.FRdataset[i][j]] < 2:
                    self.FRdataset[i][j] = '<OOV>'
                elif any(character.isdigit() for character in self.FRdataset[i][j]):
                    self.FRdataset[i][j] = '<OOV>'

        self.FRdataset = self.cleaner(self.FRdataset)
        (
            self.FRword2Index,
            self.FRindex2Word,
            self.FRvocab_size,
            self.FRvocab,
            self.FRwordFrequency
        ) = self.vocabBuilder(self.FRdataset)
        self.FRwords = list()
        for sentence in self.FRdataset:
            for word in sentence:
                self.FRwords.append(word)

        self.FRwords_indexes = [self.FRword2Index[word] for word in self.FRwords]

        print(self.FRvocab_size)

    def combineData(self):
        for idx in range(len(self.ENdataset)):
            self.combined_data.append((self.ENdataset[idx], self.FRdataset[idx]))

    def __len__(self):
        return len(self.FRdataset)

    def __getitem__(self, index):
        return (
            np.array(self.ENwords_indexes[index : index + self.sequence_length]),
            np.array(self.FRwords_indexes[index : index + self.sequence_length])
        )

def collate(data):
    X = [x[0] for x in data]
    Y = [y[1] for y in data]

    x_len = max([len(x) for x in X])
    y_len = max([len(y) for y in Y])

    padded_x = np.zeros((BATCH_SIZE, x_len))
    padded_y = np.zeros((BATCH_SIZE, y_len))

    for idx, (x, y) in enumerate(zip(X,Y)):
        padded_x[idx] = numpy.pad(x, (0,x_len - len(x)))
        padded_y[idx] = numpy.pad(y, (0,y_len - len(y))) 
    
    return (
        torch.tensor(padded_x, dtype=torch.long).t().to(device),
        torch.tensor(padded_y, dtype=torch.long).t().to(device)
    )

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout ,output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding_layer = nn.Embedding(self.input_dim, self.embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout = dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, source):
        embedding = self.dropout(self.embedding_layer(source))
        output, (state_h, state_c) = self.lstm(embedding)
        return state_h, state_c

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding_layer = nn.Embedding(output_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout = dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, source, state_h, state_c):
        source = source.unsqueeze(0)
        embedding = self.dropout(self.embedding_layer(source))
        output, (state_h, state_c) = self.lstm(embedding, (state_h, state_c))
        pred = self.fc(output.squeeze(0))
        return pred, state_h, state_c

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, source, ground_truth, force_teaching_ratio=0.5):
        # ground_truth.shape[0] = lenght of the sentence
        # ground_truth.shape[1] = batch_size
        state_h, state_c = self.encoder(source)
        outputs = torch.zeros(ground_truth.shape[0], ground_truth.shape[1], self.decoder.output_dim).to(device)
        decoder_input = ground_truth[0,:]

        for idx in range(1,ground_truth.shape[0]):
            output, state_h, state_c = self.decoder(decoder_input, state_h, state_c)
            outputs[idx] = output
            force = random.random() < force_teaching_ratio
            predicted = output.argmax(1)
            decoder_input = ground_truth[idx] if force else predicted 
        
        return outputs

def train(model, optimizer, criterion, dataloader):
    
    for epoch in range(MAX_EPOCHS):
        model.train().to(device)
        epoch_loss = 0.0
        for x,y in tqdm(dataloader):
            optimizer.zero_grad()
            pred = model(x, y)
            pred = pred[1:].reshape(-1,pred.shape[-1])
            y = y[1:].reshape(-1)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print({ 'epoch': epoch, 'loss':epoch_loss/len(dataloader) })        

class RNN(nn.Module):
    def __init__(
        self,
        dataset,
        lstm_size=128,
        n_layers=3,
        embedding_dim=128,
    ):
        super(RNN, self).__init__()
        self.vocab_size = dataset.vocab_size
        self.input_dim = lstm_size
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_size
        self.n_layers = n_layers
        self.embedding_layer = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.n_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_hidden_dim, self.vocab_size)
        self.output_dim = data.FRvocabSize

    def forward(self, x, prev_state=None):
        if prev_state == None:
            prev_state = self.init_state(1)
        embed = self.embedding_layer(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (
            torch.zeros(self.n_layers, sequence_length, self.lstm_hidden_dim).to(device),
            torch.zeros(self.n_layers, sequence_length, self.lstm_hidden_dim).to(device),
        )

def translate(text):
  model.eval()
  with torch.no_grad():
    tokens = text.split(' ')
    for idx in range(len(tokens)):
      if tokens[idx] not in data.ENvocab:
        tokens[idx] = '<OOV>'
    
    tokens = ['<SOS>'] + tokens + ['<EOS>']
    src_indexes = [data.ENword2Index[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_tensor = src_tensor.reshape(-1,1)

    output = model(src_tensor, src_tensor)
    output_dim = output.shape[-1]
    output = output.view(-1, output_dim)
    indices = torch.argmax(output,dim=1).tolist()
    return [data.FRindex2Word[x] for x in indices]


if __name__ == '__main__':
    data = Dataset("./data/ted-talks-corpus/train.en", "./data/ted-talks-corpus/train.fr", 3)
    data.combined_data = sorted(data.combined_data, key=lambda x:len(x[0]))
    dataloader = DataLoader(data, shuffle=False, collate_fn=collate, batch_size=BATCH_SIZE, drop_last=True)
    enc = Encoder(12824,EMBEDDING_SIZE, HIDDEN_DIMENSION, LAYERS, 0.5, 12824)
    dec = Decoder(15821,EMBEDDING_SIZE, HIDDEN_DIMENSION, LAYERS, 0.5)
    enc.load_state_dict(torch.load('./models/encoder_weights.pth'))
    dec.load_state_dict(torch.load('./models/decoder_weights.pth'))
    model = Seq2Seq(enc, dec)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    # train(model, optimizer, criterion, dataloader)
    model = torch.load(sys.argv[1])
    input_sentence = input("Input Sentence : ")
    print(" ".join(translate(input_sentence)[1:-1]))