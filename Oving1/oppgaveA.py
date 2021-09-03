import pandas as pd
import torch
from matplotlib import pyplot as plt


class LinearRegressionModel:

    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]],
                              requires_grad=True,
                              dtype=torch.double)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True, dtype=torch.double)

    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


if __name__ == '__main__':
    train = pd.read_csv('length_weight.csv')
    # Get the names of the columns
    # print(train.head)
    x_tensor = torch.tensor(train["# length"].to_numpy(),
                            dtype=torch.double).reshape(-1, 1)
    y_tensor = torch.tensor(train["weight"].to_numpy(),
                            dtype=torch.double).reshape(-1, 1)

    # Verify that you have gotten the right data
    # print(x_tensor[0])
    # print(y_tensor[0])

    model = LinearRegressionModel()
    optimizer = torch.optim.SGD([model.W, model.b], 0.0001)

    for epoch in range(500000):
        model.loss(x_tensor, y_tensor).backward()
        optimizer.step()
        optimizer.zero_grad()

    print("W = %s, b = %s, loss = %s" % (
        model.W, model.b, model.loss(x_tensor, y_tensor)))

    # Visualize result
    plt.plot(x_tensor, y_tensor, 'o', label='$(x^{(i)},y^{(i)})$')
    plt.xlabel('x')
    plt.ylabel('y')
    x = torch.tensor(
        [[torch.min(x_tensor)], [torch.max(x_tensor)]])
    plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = xW+b$')
    plt.legend()
    plt.show()
