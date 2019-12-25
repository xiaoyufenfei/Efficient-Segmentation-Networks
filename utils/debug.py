import sys
import traceback
import pdb

"""
This module is for debugging without modifying scripts.

By just adding `import debug` to a script which you want to debug,
automatically pdb debugger starts at the point exception raised.
After launching debugger, `from IPython import embed; embed()` enables us to run IPython.
"""


def info(exctype, value, tb):
    # we are in interactive mode or we don't have a tty-like
    # device, so we call the default hook
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        sys.__excepthook__(exctype, value, tb)
    else:
        traceback.print_exception(exctype, value, tb)
        pdb.post_mortem(tb)


sys.excepthook = info
