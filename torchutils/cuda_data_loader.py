import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


class InMemoryDataLoader(object):

    """In memory alternative to torch.utils.data.DataLoader.

    Use it whenever the transforms you apply are not stochastic, and
    your data fits in the memory. No fancy device management yet.

    """

    def __init__(self, dataset: Dataset, cuda: bool = True,
                 batch_size: int = 64, shuffle: bool = True,
                 drop_last: bool = True, keep_cpu_copy: bool = True,
                 **kwargs):

        if kwargs:
            unk_args: str = ", ".join(list(kwargs.values()))
            raise Exception(f"Keywora Arguments {unk_args:s} not supported.")

        self.dataset = dataset
        self.__length: int = len(dataset)

        self.__batch_size = batch_size if batch_size > 0 else len(dataset)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.__start_idx = self.__length

        loader = DataLoader(dataset, batch_size=self.__length, shuffle=False)
        data, target = next(iter(loader))  # data, and target are on CPU

        self.__is_cuda = cuda

        if keep_cpu_copy or not cuda:
            self.__cpu_data, self.__cpu_target = data, target
            if not cuda:
                self.data, self.target = self.__cpu_data, self.__cpu_target
        else:
            self.__cpu_data, self.__cpu_target = None, None

        if cuda:
            self.__cuda_data, self.__cuda_target = data.cuda(), target.cuda()
            self.data, self.target = self.__cuda_data, self.__cuda_target
        else:
            self.__cuda_data, self.__cuda_target = None, None

    @property
    def batch_size(self):
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self.__batch_size = batch_size if batch_size > 0 else self.__length

    @property
    def is_cuda(self):
        return self.__is_cuda

    def cpu(self, keep_cuda_copy: bool = True) -> None:
        if not self.__is_cuda:
            return

        if self.__cpu_data is None:
            self.__cpu_data = self.data.cpu()
            self.__cpu_target = self.target.cpu()

        self.data, self.target = self.__cpu_data, self.__cpu_target

        if not keep_cuda_copy:
            self.__cuda_data, self.__cuda_target = None, None

    def cuda(self, keep_cpu_copy: bool = True) -> None:
        if self.__is_cuda:
            return

        if self.__cuda_data is None:
            self.__cuda_data = self.data.cuda()
            self.__cuda_target = self.target.cuda()

        self.data, self.target = self.__cuda_data, self.__cuda_target

        if not keep_cpu_copy:
            self.__cpu_data, self.__cpu_target = None, None

    def __iter__(self):
        if self.shuffle and self.__batch_size < self.__length:
            idx = torch.randperm(self.__length)
            if self.__cpu_data is not None:
                cpu_data, cpu_target = self.__cpu_data, self.__cpu_target
            else:
                cpu_data, cpu_target = self.data.cpu(), self.target.cpu()
            if self.__is_cuda:
                self.__cuda_data.copy_(cpu_data.index_select(0, idx))
                self.__cuda_target.copy_(cpu_target.index_select(0, idx))
            else:
                self.__cpu_data = cpu_data.index_select(0, idx)
                self.__cpu_target = cpu_data.index_select(0, idx)
                self.data, self.target = self.__cpu_data, self.__cpu_target
        self.__start_idx = 0
        return self

    def __next__(self):
        start_idx = self.__start_idx
        if start_idx >= self.__length:
            raise StopIteration
        end_idx = start_idx + self.__batch_size
        if end_idx > self.__length:
            if self.drop_last:
                raise StopIteration
            else:
                end_idx = self.__length
        data = self.data[start_idx: end_idx]
        target = self.target[start_idx: end_idx]
        self.__start_idx = end_idx
        return data, target

    def __length__(self):
        return self.__length


__all__ = ["InMemoryDataLoader"]
