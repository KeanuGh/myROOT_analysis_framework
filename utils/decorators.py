import functools


def check_single_datafile(func):
    """
    Decorator to apply to src methods that require only a in single dataset.
    If no dataset name is given, and src contains only one dataset, it passes that dataset name to the method.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if (args and isinstance(args[0], str)) or 'datasets' in kwargs:
            # if dataset name is passed, just return function as-is
            return func(self, *args, **kwargs)
        elif len(self.datasets) == 1:
            # if dataset name hasn't been passed and there is only one dataset, pass it as ds_name
            return func(self, *args, **kwargs, ds_name=next(iter(self.datasets)))
        else:
            raise Exception("Must supply a dataset name when src contains more than one dataset.")
    return wrapper
