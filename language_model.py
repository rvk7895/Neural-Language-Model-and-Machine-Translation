import enum
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import math

torch.manual_seed(1)

BATCH_SIZE = 16
EPOCHS = 10
SEQUENCE_LENGTH = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_location, sequence_length):
        self.file_location = file_location
        self.sequence_length = sequence_length
        self.initialize_data()
        self.modify()

    def initialize_data(self):
        with open(self.file_location, "r") as inFile:
            data = inFile.readlines()

        tokenized_data = self.tokenizer(data)
        self.dataset = self.cleaner(tokenized_data)
        (
            self.word2Index,
            self.index2Word,
            self.vocab_size,
            self.vocab,
            self.wordFrequency
        ) = self.vocabBuilder(self.dataset)
        self.words = list()
        for sentence in self.dataset:
            for word in sentence:
                self.words.append(word)

        self.words_indexes = [self.word2Index[word] for word in self.words]

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
        for i in range(len(self.dataset)):
            for j in range(len(self.dataset[i])):
                if self.wordFrequency[self.dataset[i][j]] < 2:
                    self.dataset[i][j] = '<OOV>'
                elif any(character.isdigit() for character in self.dataset[i][j]):
                    self.dataset[i][j] = '<OOV>'
        
        self.dataset = self.cleaner(self.dataset)
        (
            self.word2Index,
            self.index2Word,
            self.vocab_size,
            self.vocab,
            self.wordFrequency
        ) = self.vocabBuilder(self.dataset)

        self.words = list()
        for sentence in self.dataset:
            for word in sentence:
                self.words.append(word)

        self.words_indexes = list()
        self.words_indexes = [self.word2Index[word] for word in self.words]

    def __len__(self):
        return len(self.words_indexes) - self.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index : index + self.sequence_length]).to(device),
            torch.tensor(
                self.words_indexes[index + 1 : index + self.sequence_length + 1]
            ).to(device),
        )

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

    def forward(self, x, prev_state):
        embed = self.embedding_layer(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (
            torch.zeros(self.n_layers, sequence_length, self.lstm_hidden_dim).to(device),
            torch.zeros(self.n_layers, sequence_length, self.lstm_hidden_dim).to(device),
        )

def calculate_probability(dataset, model, text, sequence_length=SEQUENCE_LENGTH):
    model.eval()
    text = text.lower()
    words = text.split(' ')
    sentence_probability = 0.0
    state_h, state_c = model.init_state(sequence_length)

    for i in range(0,len(words)):
        if words[i] not in dataset.vocab:
            words[i] = '<OOV>'
        

    for i in range(0, len(words)-sequence_length):
        x = torch.tensor([[dataset.word2Index[w] for w in words[i:i+sequence_length]]]).to(device)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        word = words[i+sequence_length]
        word_index = dataset.word2Index[word]
        last_word_logits = y_pred[0][-1]
        last_word_logits = nn.functional.softmax(last_word_logits)
        probability = last_word_logits[word_index].item()
        sentence_probability += 1e-10 if probability < 1e-10 else probability
    
    return sentence_probability/len(words)


if __name__ == '__main__':
    model = torch.load("./models/LM1.pth").to(device)
    dataset = Dataset("./data/europarl-corpus/train.europarl", SEQUENCE_LENGTH)
    input_sentence = input("Input Sentence : ")
    print(calculate_probability(dataset, model, input_sentence))
