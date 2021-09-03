import math

import pandas as pd
import torch
from matplotlib import pyplot as plt


class NonLinearRegressionModel():
    def __init__(self):
        self.W = torch.tensor([[0.0]],
                              requires_grad=True,
                              dtype=torch.float)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True, dtype=torch.float)

    def f(self, x):
        return 20 * torch.sigmoid((x @ self.W + self.b)) + 31

    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)

train = pd.read_csv('day_head_circumference.csv')

day_tensor = torch.tensor(train["# day"].to_numpy(),
                          dtype=torch.float).reshape(-1, 1)

head_tensor = torch.tensor(train[['head circumference']].to_numpy(),
                           dtype=torch.float).reshape(-1, 1)
# print(day_tensor[0])

model = NonLinearRegressionModel()
optimizer = torch.optim.SGD([model.W, model.b], 0.0000001)

for epoch in range(500000):
    model.loss(day_tensor, head_tensor).backward()
    optimizer.step()
    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (
    model.W, model.b, model.loss(day_tensor, head_tensor)))

# Visualize result
plt.figure('Nonlinear regression 2d')
plt.title('Predict head circumference based on age')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(day_tensor, head_tensor)
x = torch.arange(torch.min(day_tensor), torch.max(day_tensor), 1.0).reshape(-1, 1)
y = model.f(x).detach()
plt.plot(x, y, color='orange',
         label='$f(x) = 20\sigma(xW + b) + 31$ \n$\sigma(z) = \dfrac{1}{1+e^{-z}}$')

plt.legend()
plt.show()