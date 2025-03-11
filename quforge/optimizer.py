import torch

class OptimizerFactory:
    @staticmethod
    def get_optimizer(optimizer_name, *args, **kwargs):
        if optimizer_name == 'Adam':
            return torch.optim.Adam(*args, **kwargs)
        elif optimizer_name == 'SGD':
            return torch.optim.SGD(*args, **kwargs)
        else:
            raise ValueError(f"Optimizer '{optimizer_name}' is not supported")

class optim:
    Adam = staticmethod(lambda *args, **kwargs: OptimizerFactory.get_optimizer('Adam', *args, **kwargs))
    SGD = staticmethod(lambda *args, **kwargs: OptimizerFactory.get_optimizer('SGD', *args, **kwargs))
