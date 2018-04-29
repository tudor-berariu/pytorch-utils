import time
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchutils import CudaDataLoader


def main():
    from argparse import ArgumentParser
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-b", "--batch_size", dest="batch_size",
                            type=int, default=1)
    arg_parser.add_argument("-e", "--epochs", dest="epochs",
                            type=int, default=10)
    arg_parser.add_argument("-s", "--shuffle", action="store_true",
                            dest="shuffle", default=False)
    args = arg_parser.parse_args()

    batch_size = args.batch_size
    shuffle = args.shuffle

    print(f"{args.epochs:d} epochs; "
          f"batch_size={batch_size:d}; "
          f"shuffle={shuffle}")

    mean, std = -0.781000018119812, 0.33181124925613403

    dataset: Dataset = datasets.FashionMNIST(
        f'.fashion_data', train=True, download=True,
        # All applied transforms are deterministic
        transform=transforms.Compose([
            transforms.Pad((2, 2, 2, 2)),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ]))

    model = nn.Linear(32 * 32, 10).cuda()

    cuda_loader = CudaDataLoader(dataset, batch_size=batch_size,
                                 shuffle=shuffle)
    times = []
    for _ in range(args.epochs):
        start = time.time()
        for data, _ in cuda_loader:
            _ = model(Variable(data.view(-1, 32 * 32)))
        end = time.time()
        times.append(end - start)

    print(f"CudaDataLoader avg. time per epoch: {np.mean(times):5.2f}s.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=32, pin_memory=True)
    times = []
    for _ in range(args.epochs):
        start = time.time()
        for data, _ in loader:
            _ = model(Variable(data.cuda().view(-1, 32 * 32)))
        end = time.time()
        times.append(end - start)

    print(f"DataLoader avg. time per epoch: {np.mean(times):5.2f}s.")


if __name__ == "__main__":
    main()
