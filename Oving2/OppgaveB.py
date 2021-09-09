import numpy as np
import torch as torch
from matplotlib import pyplot as plt


class NandModel():
    def __init__(self):
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True, dtype=torch.float)
        self.b = torch.tensor([[0.0]], requires_grad=True, dtype=torch.float)

    def f(self, x1, x2):
        return torch.sigmoid((x1 @ self.W[0]) + (x2 @ self.W[1]) + self.b)

    def logits(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(
            self.logits(x), y)


if __name__ == '__main__':
    model = NandModel()
    optimizer = torch.optim.SGD([model.W, model.b], 0.1)

    x_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
    y_data = torch.tensor([[1], [1], [1], [0]], dtype=torch.float)


    for epoch in range(500_000):
        model.loss(x_data, y_data).backward()
        optimizer.step()
        optimizer.zero_grad()

    # Visualize result
    fig = plt.figure('Oppgave B')
    plot = fig.add_subplot(111, projection='3d')
    plt.title('NAND-operator')

    # Hva gjør denne plottinga? Aner ikke
    x1_grid, x2_grid = np.meshgrid(
        np.linspace(-0.25, 1.25, 10), np.linspace(-0.25, 1.25, 10))
    y_grid = np.empty([10, 10], dtype=np.double)
    for i in range(0, x1_grid.shape[0]):
        for j in range(0, x1_grid.shape[1]):
            tenseX = torch.tensor(float(x1_grid[i, j])).reshape(-1, 1)
            tenseY = torch.tensor(float(x2_grid[i, j])).reshape(-1, 1)
            y_grid[i, j] = model.f(tenseX, tenseY)
    plot_f = plot.plot_wireframe(x1_grid, x2_grid, y_grid, color="green")

    plot.plot(x_data[:, 0].squeeze(),
              x_data[:, 1].squeeze(),
              y_data[:, 0].squeeze(),
              'o',
              color="blue")

    plot.set_xlabel("$x_1$")
    plot.set_ylabel("$x_2$")
    plot.set_zlabel("$y$")
    plot.set_xticks([0, 1])
    plot.set_yticks([0, 1])
    plot.set_zticks([0, 1])
    plot.set_xlim(-0.25, 1.25)
    plot.set_ylim(-0.25, 1.25)
    plot.set_zlim(-0.25, 1.25)

    table = plt.table(cellText=[[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
                      colWidths=[0.1] * 3,
                      colLabels=["$x_1$", "$x_2$", "$f(x)$"],
                      cellLoc="center",
                      loc="lower right")
    plt.show()