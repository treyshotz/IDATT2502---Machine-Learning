import pandas as pd
import torch
from matplotlib import pyplot as plt


class LinearRegressionModel:

    def __init__(self):
        # Model variables
        self.W = torch.rand((2, 1), requires_grad=True, dtype=torch.double)
        self.b = torch.rand((1, 1), requires_grad=True, dtype=torch.double)

    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


train = pd.read_csv('day_length_weight.csv')

# print(train.head)

day_tensor = torch.tensor(train["# day"].to_numpy(),
                          dtype=torch.double).reshape(-1, 1)
length_weight_tensor = torch.tensor(train[['length', 'weight']].to_numpy(),
                                    dtype=torch.double).reshape(-1, 2)
# print(day_tensor[0])
# print(length_weight_tensor[0])

model = LinearRegressionModel()
optimizer = torch.optim.SGD([model.W, model.b], 0.0001)

for epoch in range(500000):
    model.loss(length_weight_tensor, day_tensor).backward()
    optimizer.step()
    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (
    model.W, model.b, model.loss(length_weight_tensor, day_tensor)))

length = length_weight_tensor.t()[0]
weight = length_weight_tensor.t()[1]

fig = plt.figure('Linear regression 3d')
ax = fig.add_subplot(projection='3d',
                     title="Model for predicting days lived by weight and length")
# Plot
ax.scatter(length.numpy(), weight.numpy(), day_tensor.numpy(),
           label='$(x^{(i)},y^{(i)}, z^{(i)})$')
ax.scatter(length.numpy(), weight.numpy(),
           model.f(length_weight_tensor).detach().numpy(),
           label='$\\hat y = f(x) = xW+b$', color="orange")
ax.legend()
plt.show()
