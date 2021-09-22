import numpy as np
import torch
import torch.nn as nn


class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, encoding_size)  # 128 is the state size

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 1,
                                 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self,
        x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (
            self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x,
        y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


char_encodings = [
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # ' '
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'a'
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'c'
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'd'
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'f'
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],  # 'h'
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],  # 'l'
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 'm'
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # 'n'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],  # 'o'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],  # 'r'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],  # 's'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],  # 't'
]
encoding_size = len(char_encodings)
#                 0    1    2    3    4    5    6    7    8    9   10   11   12
index_to_char = [' ', 'a', 'c', 'd', 'f', 'h', 'l', 'm', 'n', 'o', 'r', 's',
                 't']
index_to_emojis = ['🎩', '🐀', '🐈', '🏢', '🚗', '👶']

emojis = np.eye(len(index_to_emojis))

x_train = torch.tensor(
    [[[char_encodings[5]], [char_encodings[1]], [char_encodings[12]],
      [char_encodings[0]]],  # hat
     [[char_encodings[10]], [char_encodings[1]], [char_encodings[12]],
      [char_encodings[0]]],  # rat
     [[char_encodings[2]], [char_encodings[1]], [char_encodings[12]],
      [char_encodings[0]]],  # cat
     [[char_encodings[4]], [char_encodings[6]], [char_encodings[1]],
      [char_encodings[12]]],  # flat
     [[char_encodings[2]], [char_encodings[1]], [char_encodings[10]],
      [char_encodings[0]]],  # car
     [[char_encodings[11]], [char_encodings[9]], [char_encodings[8]],
      [char_encodings[0]]]  # son
     ])

y_train = torch.tensor(
    [[emojis[0], emojis[0], emojis[0], emojis[0]],
     [emojis[1], emojis[1], emojis[1], emojis[1]],
     [emojis[2], emojis[2], emojis[2], emojis[2]],
     [emojis[3], emojis[3], emojis[3], emojis[3]],
     [emojis[4], emojis[4], emojis[4], emojis[4]],
     [emojis[5], emojis[5], emojis[5], emojis[5]]])

print(x_train.shape)
print(y_train.shape)

model = LongShortTermMemoryModel(encoding_size)

optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
for epoch in range(500):
    for i in range(x_train.size()[0]):
        model.reset()
        model.loss(x_train[i], y_train[i]).backward()
        optimizer.step()
        optimizer.zero_grad()


    def generate_emoji(string):
        y = -1
        model.reset()
        for i in range(len(string)):
            char_index = index_to_char.index(string[i])
            y = model.f(torch.tensor([[char_encodings[char_index]]],
                                     dtype=torch.float))
        print(index_to_emojis[y.argmax(1)])

generate_emoji('rt')
generate_emoji('ht')
generate_emoji('so')
generate_emoji('fat')
generate_emoji('cr')
generate_emoji('ct')
