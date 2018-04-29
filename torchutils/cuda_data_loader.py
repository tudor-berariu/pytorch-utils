import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


class CudaDataLoader(object):

    """GPU alternative to torch.utils.data.DataLoader.

    Use it whenever the transforms you apply are not stochastic, and
    your data fits in the GPU memory. No fancy device management yet.

    """

    def __init__(self, dataset: Dataset,
                 batch_size: int = 64, shuffle: bool = True,
                 drop_last: bool = True, keep_cpu_copy: bool = True,
                 **kwargs):

        if kwargs:
            unk_args: str = ", ".join(list(kwargs.values()))
            raise Exception(f"Keywora Arguments {unk_args:s} not supported.")

        self.__length: int = len(dataset)
        self.keep_cpu_copy = keep_cpu_copy
        self.__batch_size = batch_size if batch_size > 0 else len(dataset)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.__start_idx = self.__length

        loader = DataLoader(dataset, batch_size=self.__length, shuffle=False)
        data, target = next(iter(loader))
        if keep_cpu_copy:
            self.__cpu_data, self.__cpu_target = data, target
        self.data, self.target = data.cuda(), target.cuda()

    @property
    def batch_size(self):
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self.__batch_size = batch_size if batch_size > 0 else self.__length

    def __iter__(self):
        if self.shuffle and self.__batch_size < self.__length:
            idx = torch.randperm(self.__length)
            if self.keep_cpu_copy:
                cpu_data, cpu_target = self.__cpu_data, self.__cpu_target
            else:
                cpu_data, cpu_target = self.data.cpu(), self.target.cpu()
                del self.data, self.target
            self.data = cpu_data.index_select(0, idx).cuda()
            self.target = cpu_target.index_select(0, idx).cuda()
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


__all__ = ["CudaDataLoader"]
