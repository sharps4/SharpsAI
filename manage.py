import re
import sys
import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from collections import Counter
from torch.utils.data import DataLoader
# from generator import utils


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model class: neural network 
class Model(nn.Module):
    def __init__(self, n_vocab):
        super(Model, self).__init__()
        self.lstm_size = 270
        self.embedding_dim = 220
        self.num_layers = 3
        self.n_vocab = n_vocab
        self.dropout_rate = 0.2

        # Define the layers 
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=self.dropout_rate
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)


    def forward(self, x, prev_state): #temperature=0.8
        # Perform forward propagation
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)

        logits.scatter_(-1, x.unsqueeze(-1), float('-inf'))
        
        return logits, state

    def init_state(self, sequence_length):
        # Initilize the state
        return (
            torch.zeros(self.num_layers, sequence_length, self.lstm_size).data.to(device),
            torch.zeros(self.num_layers, sequence_length, self.lstm_size).data.to(device),
       )

# Dataset class: load the data
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_path,
        sequence_length,
        num_line,
    ):
        self.file_path = file_path
        self.num_line = num_line
        self.sequence_length = sequence_length
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()
        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
        self.words_indexes = [self.word_to_index[w] for w in self.words]
        self.words_indexes = torch.tensor(self.words_indexes).data.to(device)
        
    def load_words(self):
        # Load and clean the text data
        with open(self.file_path, encoding='utf-8', errors='ignore') as f:
            data = f.read()
        if self.num_line:
            text = self.text_cleaner(data[:self.num_line])
        else:
            text = self.text_cleaner(data)
        return text.split()
    
    def text_cleaner(self, text):
        # Clean the text data
        text = text.lower()
        newString = re.sub(r"'s\b","", text)
        # keep letters, digits, and punctuation
        newString = re.sub("[^a-zA-ZÀ-ÿ0-9.,!?]", " ", newString) 
        long_words=[]
        for i in newString.split():              
            long_words.append(i)
        return (" ".join(long_words)).strip()
    
    def get_uniq_words(self):
        # Count word occurrences and return the unique words in descending order of frequency
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)
    
    def __len__(self):
        # Return the length of the dataset
        return len(self.words_indexes) - self.sequence_length
    
    def __getitem__(self, index):
        # Get a sample from the dataset, consisting of input and target sequences
        return (
            self.words_indexes[index:index+self.sequence_length],
            self.words_indexes[index+1:index+self.sequence_length+1],
        )  

def text_cleaner(text):
        text = text.lower()
        newString = re.sub(r"'s\b","", text)
        newString = re.sub("[^a-zA-ZÀ-ÿ0-9.,!?]", " ", newString)
        long_words=[]
        for i in newString.split():              
            long_words.append(i)
            print(long_words)
        return (" ".join(long_words)).strip()

def predict(dataset, model, text, next_words=20, temperature=0.8):
    # Generate new text
    text = text_cleaner(text)
    words = text.split()
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]]).to(device)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().to('cpu').numpy()

        # Apply temperature
        p = np.power(p, 1.0 / temperature)
        p /= np.sum(p)

        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return words[1:]


def generate_text(text, length):
    # Generate text using a trained model
    try:
        return ' '.join(predict(dataset, loded_model, text, next_words=length))
    except:
        return "Impossible de générer du texte pour la séquence donnée, certains mots ne sont pas présents dans les données d'entrainement." 


file_path = "data/new_fr_data_1.txt"
sequence_length = 15
dataset = Dataset(file_path, sequence_length, num_line=1_300_000)
n_vocab = len(dataset.uniq_words)
model_path = "model_dict4.pkl"




"""Django's command-line utility for administrative tasks."""
import os
import sys



def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TextGeneratorBot.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc


    execute_from_command_line(sys.argv)

if __name__ == "__main__":
    print("Loading model...")
    loded_model = Model(n_vocab).to(device)
    loded_model.load_state_dict(torch.load(model_path))
    loded_model.eval()

    main()