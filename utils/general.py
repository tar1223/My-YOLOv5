import inspect
import logging
import os
import platform
from pathlib import Path
from typing import Optional

from utils import emojis

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

LOGGING_NAME = 'yolov5'


def set_logging(name=LOGGING_NAME, verbose=True):
    rank = int(os.getenv('RANK', -1))
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format": "%(message)s'
            }
        },
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level
            }
        },
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False
            }
        }
    })


set_logging(LOGGING_NAME)
LOGGER = logging.getLogger(LOGGING_NAME)
if platform.system() == 'Windows':
    for fn in LOGGER.info, LOGGER.warning:
        setattr(LOGGER, fn.__name__, lambda x: fn(emojis(x)))


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
