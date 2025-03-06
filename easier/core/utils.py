# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import logging
import random
import string

LOGGING_TRACE = 5

# We cannot decide
ompi_rank = os.environ.get("OMPI_COMM_WORLD_RANK", None)
torchrun_rank = os.environ.get("RANK", None)
logger = logging.getLogger(f"Rank{torchrun_rank or ompi_rank}")
# DEBUG, INFO, WARNING, ERROR, CRITICAL
# NOTE environ variable EASIER_LOG_LEVEL can be specified on `torchrun` process
# and will be inherited by all worker processes.
log_level = os.environ.get("EASIER_LOG_LEVEL", logging.INFO)
try:
    log_level = int(log_level)
except ValueError:
    pass
logger.setLevel(log_level)
handler = logging.StreamHandler()  # FileHandler, StreamHandler

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

handler.setFormatter(formatter)
logger.addHandler(handler)


class EasierJitException(Exception):
    pass


def get_random_str(length=8):
    return "".join(
        random.choice(string.ascii_lowercase + string.digits)
        for _ in range(length))
