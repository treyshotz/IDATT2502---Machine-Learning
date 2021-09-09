import torch as torch
from matplotlib import pyplot as plt


class NotModel():
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True, dtype=torch.float)
        self.b = torch.tensor([[0.0]], requires_grad=True, dtype=torch.float)

    def f(self, x):
        return torch.sigmoid(self.logits(x))

    def logits(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(
            self.logits(x), y)


if __name__ == '__main__':
    model = NotModel()
    optimizer = torch.optim.SGD([model.W, model.b], 0.1)

    x_data = torch.tensor([[0], [1]], dtype=torch.float)
    y_data = torch.tensor([[1], [0]], dtype=torch.float)

    for epoch in range(50_000):
        model.loss(x_data, y_data).backward()
        optimizer.step()
        optimizer.zero_grad()

    print("W = %s, b = %s, loss = %s" % (
        model.W, model.b, model.loss(x_data, y_data)))

    plt.figure('oppgaveA')
    plt.title('NOT')
    plt.table(cellText=[[0, 1], [1, 0]],
              colWidths=[0.1] * 3,
              colLabels=["$x$", "$f(x)$"],
              cellLoc="center",
              loc="lower left")
    plt.scatter(x_data, y_data)
    plt.xlabel('x')
    plt.ylabel('y')
    x = torch.arange(0.0, 1.0, 0.001).reshape(-1, 1)
    y = model.f(x).detach()
    plt.plot(x, y, color="orange")
    plt.show()
