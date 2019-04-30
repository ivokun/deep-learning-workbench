###
# From https://github.com/bentrevett/pytorch-sentiment-analysis
###
import torch
from torchtext import data
from torchtext import datasets
import torch.nn as nn
import torch.optim as optim
import time
import random

from nltk import FreqDist

class LSTM(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
    self.lstm = nn.LSTM(embedding_dim,
                        hidden_dim,
                        num_layers=n_layers,
                        bidirectional=bidirectional,
                        dropout=dropout)
    self.fc = nn.Linear(hidden_dim * 2, output_dim)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, text, text_lengths):
    embedded = self.dropout(self.embedding(text))
    packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
    packed_output, (hidden, cell) = self.lstm(packed_embedded)
    output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
    hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
    return self.fc(hidden.squeeze(0))


SEED = 1234
MAX_VOCAB_SIZE = 25_000
BATCH_SIZE = 64

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy', include_lengths = True)
LABEL = data.LabelField(dtype = torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state = random.seed(SEED))

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples:{len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

print(vars(train_data.examples[0]))

TEXT.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE)
                #  vectors = 'glove.6B.100d',
                #  unk_init = torch.Tensor.normal_)
LABEL.build_vocab(train_data)

print(f'Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}')
print(f'Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}')

print(TEXT.vocab.freqs.most_common(20))
print(TEXT.vocab.itos[:10])
print(LABEL.vocab.stoi)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
#   (train_data, valid_data, test_data),
#   batch_size = BATCH_SIZE,
#   sort_within_batch = True,
#   device = device)

# INPUT_DIM = len(TEXT.vocab)
# EMBEDDING_DIM = 100
# HIDDEN_DIM = 256
# OUTPUT_DIM = 1
# N_LAYERS = 2
# BIDIRECTIONAL = False
# DROPOUT = 0.5
# PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

# model = LSTM(INPUT_DIM,
#              EMBEDDING_DIM,
#              HIDDEN_DIM,
#              OUTPUT_DIM,
#              N_LAYERS,
#              BIDIRECTIONAL,
#              DROPOUT,
#              PAD_IDX)

# def count_parameters(model):
#   return sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f'The model has {count_parameters(model):,} trainable parameters')

# # pretrained_embeddings = TEXT.vocab.vectors
# # print(pretrained_embeddings.shape)

# # model.embedding.weight.data.copy_(pretrained_embeddings)

# UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
# model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
# model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

# optimizer = optim.Adam(model.parameters())
# criterion = nn.BCEWithLogitsLoss()

# model = model.to(device)
# criterion = criterion.to(device)

# def binary_accuracy(preds, y):
#   """
#   Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
#   """

#   #round predictions to the closes integer
#   rounded_preds = torch.round(torch.sigmoid(preds))
#   correct = (rounded_preds == y).float()
#   acc = correct.sum() / len(correct)
#   return acc

# def train(model, iterator, optimizer, criterion):
  
#   epoch_loss = 0
#   epoch_acc = 0

#   model.train()

#   for batch in iterator:
#     optimizer.zero_grad()
#     text, text_lengths = batch.text
#     predictions = model(text, text_lengths).squeeze(1)
#     loss = criterion(predictions, batch.label)
#     acc = binary_accuracy(predictions, batch.label)

#     loss.backward()
#     optimizer.step()

#     epoch_loss += loss.item()
#     epoch_acc  += acc.item()
  
#   return epoch_loss / len(iterator), epoch_acc / len(iterator)

# def evaluate(model, iterator, criterion):
#   epoch_loss = 0
#   epoch_acc = 0

#   model.eval()

#   with torch.no_grad():
#     for batch in iterator:
#       text, text_lengths = batch.text
#       predictions = model(text, text_lengths).squeeze(1)
#       loss = criterion(predictions, batch.label)
#       acc = binary_accuracy(predictions, batch.label)

#       epoch_loss += loss.item()
#       epoch_acc += acc.item()
  
#   return epoch_loss / len(iterator), epoch_acc / len(iterator)

# def epoch_time(start_time, end_time):
#   elapsed_time = end_time - start_time
#   elapsed_mins = int(elapsed_time / 60)
#   elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
#   return elapsed_mins, elapsed_secs

# N_EPOCHS = 5

# best_valid_loss = float('inf')

# print("="*10)
# print("Train Start")
# print("="*10)


# for epoch in range(N_EPOCHS):
#   start_time = time.time()
#   train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
#   valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

#   end_time = time.time()

#   epoch_mins, epoch_secs = epoch_time(start_time, end_time)

#   if valid_loss < best_valid_loss:
#     best_valid_loss = valid_loss
#     torch.save(model.state_dict(), 'lstm-model2.pt')
  
#   print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
#   print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
#   print(f'\t Val. Loss: {valid_loss:.3f} | Val. Acc : {valid_acc*100:.2f}%')

# print("="*10)
# print("Train Finish")
# print("="*10)

# model.load_state_dict(torch.load('lstm-model2.pt'))
# test_loss, test_acc = evaluate(model, test_iterator, criterion)
# print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
