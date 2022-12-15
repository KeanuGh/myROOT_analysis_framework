import io
import os
import sys
from contextlib import contextmanager
from functools import wraps
from typing import Callable, TextIO


def check_single_dataset(func) -> Callable:
    """
    Decorator to apply to src methods that take only a single dataset as their argument.
    If no dataset name is given, and src contains only one dataset, it passes that dataset name to the method.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Callable:
        if (args and isinstance(args[0], str)) or "ds_name" in kwargs:
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
        elif "datasets" in kwargs:
            datasets = kwargs["datasets"]
            kwargs.pop("datasets")
        else:
            datasets = None

        if isinstance(datasets, str):
            # return apply to just the one function
            if datasets not in self.datasets:
                raise ValueError(f"No dataset '{datasets}' in analysis '{self.name}'")
            func(self, *args, **kwargs, datasets=datasets)

        elif hasattr(datasets, "__iter__"):
            # apply to each dataset in iterable
            for dataset in datasets:
                if not isinstance(dataset, str):
                    raise TypeError(
                        "Iterable dataset argument must be a string or iterable containing only strings"
                    )
                if dataset not in self.datasets:
                    raise ValueError(f"No dataset '{dataset}' in analysis '{self.name}'")
                func(self, *args, **kwargs, datasets=dataset)

        elif datasets is None:
            # apply to all datasets
            for dataset in self.datasets:
                func(self, *args, **kwargs, datasets=dataset)

    return wrapper


# https://stackoverflow.com/questions/24277488/in-python-how-to-capture-the-stdout-from-a-c-shared-library-to-a-variable
@contextmanager
def redirect_stdout(out_stream: TextIO = None, in_stream: TextIO = None):
    """Capture C/C++ standard output"""
    if out_stream is None:
        out_stream = io.StringIO()

    if in_stream is None:
        in_stream = sys.stdout

    og_streamfd = in_stream.fileno()
    captured_output = ""
    # Create a pipe so the stream can be captured:
    pipe_out, pipe_in = os.pipe()
    # Save a copy of the stream:
    streamfd = os.dup(og_streamfd)

    try:
        # Replace the original stream with our write pipe:
        os.dup2(pipe_in, og_streamfd)
        # yield string that will contained captured output
        yield out_stream

    finally:
        # Print escape character to make the readOutput method stop:
        escape_char = "\b"
        in_stream.write(escape_char)
        # Flush the stream to make sure all our data goes in before the escape character:
        in_stream.flush()

        # Read the stream data
        while True:
            char = os.read(pipe_out, 1).decode(in_stream.encoding)
            if not char or escape_char in char:
                break
            captured_output += char
        out_stream.write(captured_output)

        # Close the pipe:
        os.close(pipe_in)
        os.close(pipe_out)
        # Restore the original stream:
        os.dup2(streamfd, og_streamfd)
        # Close the duplicate stream:
        os.close(streamfd)
