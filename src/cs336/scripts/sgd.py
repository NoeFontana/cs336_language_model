import torch

from cs336.optim.sgd import SGD

if __name__ == "__main__":
    for lr in (1e-1, 1e1, 1e2, 1e3):
        print(f"\nLearning rate: {lr}")
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=lr)

        for _ in range(100):
            opt.zero_grad()
            loss = (weights**2).mean()
            print(loss.cpu().item())
            loss.backward()
            opt.step()
