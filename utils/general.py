import inspect
import logging
from pathlib import Path
from typing import Optional

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

LOGGING_NAME = 'yolov5'
LOGGER = logging.getLogger(LOGGING_NAME)


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    x = inspect.currentframe().f_back
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix('')
    except ValueError:
        file = Path(file).stem
    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    LOGGER.info(colorstr(s) + ', '.join(f'{k}={v}' for k, v in args.items()))


def get_latest_run():
    pass


def check_git_status():
    pass


def check_requirements():
    pass


def check_yaml():
    pass


def check_file():
    pass


def colorstr():
    pass


def print_mutation():
    pass


def increment_path():
    pass
