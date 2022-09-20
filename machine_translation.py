import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy
import spacy
import numpy as np
import random
from tqdm import tqdm
import sys

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

print("Loading Spacy Models")
en_spacy = spacy.load('en_core_web_sm')
fr_spacy = spacy.load('fr_core_news_sm')

BATCH_SIZE = 16
MAX_EPOCHS = 10
EMBEDDING_SIZE = 16
HIDDEN_DIMENSION = 128
LAYERS = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Data(Dataset):
    def __init__(self, en_location, fr_location):
        self.en_location = en_location
        self.fr_location = fr_location
        self.corpusSize = 0
        self.processed_en_dataset = list()
        self.processed_fr_dataset = list()
        self.ENvocab = set()
        self.ENword2Index = dict()
        self.ENindex2Word = list()
        self.ENwordFrequency = dict()
        self.ENvocabSize = 1
        self.FRvocab = set()
        self.FRword2Index = dict()
        self.FRindex2Word = list()
        self.FRwordFrequency = dict()
        self.FRvocabSize = 1
        self.load_data()
        self.preprocessor()
        self.vocabBuilder()
        self.modifier()
        self.combined_data = list()
        self.combineData()
    
    def load_data(self):
        with open(self.en_location, 'r') as inFile:
            self.en_dataset = inFile.readlines()
            self.en_dataset = self.en_dataset[:200000]
        
        with open(self.fr_location, 'r') as inFile:
            self.fr_dataset = inFile.readlines()
            self.fr_dataset = self.fr_dataset[:200000]
        self.corpusSize = len(self.en_dataset)
    
    def english_tokenizer(self, text):
        return [tok.text for tok in en_spacy.tokenizer(text)]

    def french_tokenizer(self, text):
        return [tok.text for tok in fr_spacy.tokenizer(text)]

    def cleaner(self,sentence):
        """
            replacing !,?,. with . and removing other punctuations
            
            Arguments:
                tokenized corpuse (list)

            Returns:
                cleaned corpus (list)
        """
        import string

        cleaned_corpus = list()

        new_sentence = list()
        for token in sentence:
            if token in string.punctuation or token == '\n':
                continue
            else:
                new_sentence.append(token)


        return new_sentence
    
    def vocabBuilder(self):
        print("Building English Vocabulary")
        for sentence in self.processed_en_dataset:
            for word in sentence:
                self.ENvocab.add(word)
                if word not in self.ENword2Index:
                    self.ENword2Index[word] = self.ENvocabSize
                    self.ENindex2Word.append(word)
                    self.ENwordFrequency[word] = 1
                    self.ENvocabSize += 1
                
                else:
                    self.ENwordFrequency[word] += 1
        print("Building French Vocabulary")
        for sentence in self.processed_fr_dataset:
            for word in sentence:
                self.FRvocab.add(word)
                if word not in self.FRword2Index:
                    self.FRword2Index[word] = self.FRvocabSize
                    self.FRindex2Word.append(word)
                    self.FRwordFrequency[word] = 1
                    self.FRvocabSize += 1
                
                else:
                    self.FRwordFrequency[word] += 1
    
    def preprocessor(self):
        for sentence in self.en_dataset:
            tokenized_sentence = self.english_tokenizer(sentence)
            cleaned_sentence = self.cleaner(tokenized_sentence)
            normalized_sentence = ['<SOS>']
            for token in cleaned_sentence:
                normalized_sentence.append(token.lower())
            normalized_sentence = normalized_sentence + ['<EOS>']

            self.processed_en_dataset.append(normalized_sentence)

        for sentence in self.fr_dataset:
            tokenized_sentence = self.french_tokenizer(sentence)
            cleaned_sentence = self.cleaner(tokenized_sentence)
            normalized_sentence = ['<SOS>']
            for token in cleaned_sentence:
                normalized_sentence.append(token.lower())
            normalized_sentence = normalized_sentence + ['<EOS>']

            self.processed_fr_dataset.append(normalized_sentence)
    
    def modifier(self):
        for i in range(self.corpusSize):
            for j in range(1, len(self.processed_en_dataset[i]) - 1):
                if self.ENwordFrequency[self.processed_en_dataset[i][j]] < 2:
                    self.processed_en_dataset[i][j] = '<OOV>'
        
        for i in range(self.corpusSize):
            for j in range(1, len(self.processed_fr_dataset[i]) - 1):
                if self.FRwordFrequency[self.processed_fr_dataset[i][j]] < 2:
                    self.processed_fr_dataset[i][j] = '<OOV>'

        self.ENvocab = set()
        self.ENword2Index = dict()
        self.ENindex2Word = list()
        self.ENwordFrequency = dict()
        self.ENvocabSize = 1
        self.FRvocab = set()
        self.FRword2Index = dict()
        self.FRindex2Word = list()
        self.FRwordFrequency = dict()
        self.FRvocabSize = 1
        
        print("Rebuilding English Vocabulary")
        for sentence in self.processed_en_dataset:
            for word in sentence:
                self.ENvocab.add(word)
                if word not in self.ENword2Index:
                    self.ENword2Index[word] = self.ENvocabSize
                    self.ENindex2Word.append(word)
                    self.ENwordFrequency[word] = 1
                    self.ENvocabSize += 1
                
                else:
                    self.ENwordFrequency[word] += 1

        print("Rebuilding French Vocabulary")
        for sentence in self.processed_fr_dataset:
            for word in sentence:
                self.FRvocab.add(word)
                if word not in self.FRword2Index:
                    self.FRword2Index[word] = self.FRvocabSize
                    self.FRindex2Word.append(word)
                    self.FRwordFrequency[word] = 1
                    self.FRvocabSize += 1
                
                else:
                    self.FRwordFrequency[word] += 1
    
    def combineData(self):
        for idx in range(self.corpusSize):
            self.combined_data.append((self.processed_en_dataset[idx], self.processed_fr_dataset[idx]))

    def __len__(self):
        return self.corpusSize
    
    def __getitem__(self, index):
        return (
            np.array([self.ENword2Index[word] for word in self.combined_data[index][0]]),
            np.array([self.FRword2Index[word] for word in self.combined_data[index][1]])
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
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding_layer = nn.Embedding(self.input_dim, self.embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout = dropout)

    def forward(self, input):
        embedding = self.dropout(self.embedding_layer(input))
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
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, state_h, state_c):
        input = input.unsqueeze(0)
        embedding = self.dropout(self.embedding_layer(input))
        output, (state_h, state_c) = self.lstm(embedding, (state_h, state_c))
        pred = self.linear(output.squeeze(0))
        return pred, state_h, state_c

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, input, ground_truth, force_teaching_ratio=0.5):
        # ground_truth.shape[0] = lenght of the sentence
        # ground_truth.shape[1] = batch_size
        state_h, state_c = self.encoder(input)
        outputs = torch.zeros(ground_truth.shape[0], ground_truth.shape[1], self.decoder.output_dim).to(device)
        decoder_input = ground_truth[0,:]

        for idx in range(1,ground_truth.shape[0]):
            output, state_h, state_c = self.decoder(decoder_input, state_h, state_c)
            outputs[idx] = output
            force = random.random() < force_teaching_ratio
            predicted = output.argmax(1)
            decoder_input = ground_truth[idx] if force else predicted 
        
        return outputs

def translate(text):
  model.eval()
  with torch.no_grad():
    tokens = data.english_tokenizer(text)
    tokens = data.cleaner(tokens)
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
    print("Loading Data")
    data = Data('./data/ted-talks-corpus/train.en','./data/ted-talks-corpus/train.fr')
    data.combined_data = sorted(data.combined_data, key=lambda x:len(x[0]))
    dataloader = DataLoader(data, shuffle=False, collate_fn=collate, batch_size=BATCH_SIZE, drop_last=True)
    model_location = sys.argv[1]
    model = torch.load(model_location).to(device)
    input_sentence = input("Input Sentence : ")
    print(" ".join(translate(input_sentence)[1:]))