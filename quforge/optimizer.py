import torch


class OptimizerFactory:
    """
    A factory class for creating PyTorch optimizers.

    This class provides a static method to obtain a PyTorch optimizer based on the given optimizer name.
    It supports commonly used optimizers such as Adam and SGD. If an unsupported optimizer name is provided,
    a ValueError is raised.

    Methods:
        get_optimizer(optimizer_name, *args, **kwargs):
            Returns a PyTorch optimizer corresponding to the given optimizer_name using the provided
            arguments and keyword arguments.

    Example:
        >>> optimizer = OptimizerFactory.get_optimizer('Adam', model.parameters(), lr=0.001)
    """

    @staticmethod
    def get_optimizer(optimizer_name, *args, **kwargs):
        if optimizer_name == "Adam":
            return torch.optim.Adam(*args, **kwargs)
        if optimizer_name == "SGD":
            return torch.optim.SGD(*args, **kwargs)
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported")


class optim:
    """
    A simple wrapper for PyTorch optimizers.

    This class provides static methods that serve as shorthand for creating PyTorch optimizers using the
    OptimizerFactory. Currently supported optimizers include:

        - Adam
        - SGD

    Usage:
        >>> optimizer = qf.optim.Adam(model.parameters(), lr=0.001)
        >>> optimizer = qf.optim.SGD(model.parameters(), lr=0.01)
    """

    Adam = staticmethod(
        lambda *args, **kwargs: OptimizerFactory.get_optimizer("Adam", *args, **kwargs)
    )
    SGD = staticmethod(
        lambda *args, **kwargs: OptimizerFactory.get_optimizer("SGD", *args, **kwargs)
    )
