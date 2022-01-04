from functools import wraps
from typing import Callable


def check_single_dataset(func) -> Callable:
    """
    Decorator to apply to src methods that take only a single dataset as their argument.
    If no dataset name is given, and src contains only one dataset, it passes that dataset name to the method.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Callable:
        if (args and isinstance(args[0], str)) or 'ds_name' in kwargs:
            # if dataset name is passed, just return function as-is
            return func(self, *args, **kwargs)
        elif len(self.datasets) == 1:
            # if dataset name hasn't been passed and there is only one dataset, pass it as ds_name
            return func(self, *args, **kwargs, ds_name=next(iter(self.datasets)))
        else:
            raise Exception("Must supply a dataset name when src contains more than one dataset.")

    return wrapper


def handle_dataset_arg(func: Callable) -> Callable:
    """
    if 'datasets' argument is not given, run over all datasets.
    If a string is passed look for named dataset
    If list of strings is passed, run over given datasets

    FUNCTION SHOULD NOT RETURN ANYTHING
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs) -> None:
        # get datasets argument from function
        if args:
            datasets = args[0]
            args = args[1:]
        elif 'datasets' in kwargs:
            datasets = kwargs['datasets']
            kwargs.pop('datasets')
        else:
            datasets = None

        if isinstance(datasets, str):
            # return apply to just the one function
            if datasets not in self.datasets:
                raise ValueError(f"No dataset '{datasets}' in analysis '{self.name}'")
            func(self, *args, **kwargs)

        elif hasattr(datasets, '__iter__'):
            # apply to each dataset in iterable
            for dataset in datasets:
                if not isinstance(dataset, str):
                    raise TypeError("Iterable dataset argument must be a string or iterable containing only strings")
                if dataset not in self.datasets:
                    raise ValueError(f"No dataset '{dataset}' in analysis '{self.name}'")
                func(self, *args, **kwargs, datasets=dataset)

        elif datasets is None:
            # apply to all datasets
            for dataset in self.datasets:
                func(self, *args, **kwargs, datasets=dataset)

    return wrapper
