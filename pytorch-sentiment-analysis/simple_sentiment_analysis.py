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

class RNN(nn.Module):
  def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
    super().__init__()
    self.embedding = nn.Embedding(input_dim, embedding_dim)
    self.rnn = nn.RNN(embedding_dim, hidden_dim)
    self.fc = nn.Linear(hidden_dim, output_dim)
  
  def forward(self, text):
    embedded = self.embedding(text)
    output, hidden = self.rnn(embedded)
    assert torch.equal(output[-1,:,:], hidden.squeeze(0))
    return self.fc(hidden.squeeze(0))

SEED = 1234
MAX_VOCAB_SIZE = 25_000
BATCH_SIZE = 64

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy')
LABEL = data.LabelField(dtype = torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state = random.seed(SEED))

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples:{len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

# print(vars(train_data.examples[0]))

TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
LABEL.build_vocab(train_data)

print(f'Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}')
print(f'Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}')

print(TEXT.vocab.freqs.most_common(20))
print(TEXT.vocab.itos[:10])
print(LABEL.vocab.stoi)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
  (train_data, valid_data, test_data),
  batch_size = BATCH_SIZE,
  device = device)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
LEARNING_RATE = 1e-3

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

def binary_accuracy(preds, y):
  """
  Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
  """

  #round predictions to the closes integer
  rounded_preds = torch.round(torch.sigmoid(preds))
  correct = (rounded_preds == y).float()
  acc = correct.sum() / len(correct)
  return acc

def train(model, iterator, optimizer, criterion):
  
  epoch_loss = 0
  epoch_acc = 0

  model.train()

  for batch in iterator:
    optimizer.zero_grad()
    predictions = model(batch.text).squeeze(1)
    loss = criterion(predictions, batch.label)
    acc = binary_accuracy(predictions, batch.label)

    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()
    epoch_acc  += acc.item()
  
  return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
  epoch_loss = 0
  epoch_acc = 0

  model.eval()

  with torch.no_grad():
    for batch in iterator:
      predictions = model(batch.text).squeeze(1)
      loss = criterion(predictions, batch.label)
      acc = binary_accuracy(predictions, batch.label)

      epoch_loss += loss.item()
      epoch_acc += acc.item()
  
  return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

N_EPOCHS = 5

best_valid_loss = float('inf')

print("="*10)
print("Train Start")
print("="*10)


for epoch in range(N_EPOCHS):
  start_time = time.time()
  train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
  valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

  end_time = time.time()

  epoch_mins, epoch_secs = epoch_time(start_time, end_time)

  if valid_loss < best_valid_loss:
    best_valid_loss = valid_loss
    torch.save(model.state_dict(), 'simple-model.pt')
  
  print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
  print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
  print(f'\t Val. Loss: {valid_loss:.3f} | Val. Acc : {valid_acc*100:.2f}%')

print("="*10)
print("Train Finish")
print("="*10)

model.load_state_dict(torch.load('simple-model.pt'))
test_loss, test_acc = evaluate(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
