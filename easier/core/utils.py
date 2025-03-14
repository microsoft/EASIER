# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import logging
import random
import string

logger: logging.Logger = logging.getLogger("easier.init")

# DEBUG, INFO, WARNING, ERROR, CRITICAL
# NOTE environ variable EASIER_LOG_LEVEL can be specified on `torchrun` process
# and will be inherited by all worker processes.
logger.setLevel(os.environ.get("EASIER_LOG_LEVEL", logging.INFO))
handler = logging.StreamHandler()  # FileHandler, StreamHandler

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

handler.setFormatter(formatter)
logger.addHandler(handler)


def init_logger(rank: int):
    """
    Due to multiple launching methods (torchrun or mpirun) may exist
    hierarchically on the cloud environment, we need easier.init to provide
    a reliable rank.
    """
    logger.name = f"Rank{rank}"


class EasierJitException(Exception):
    pass


def get_random_str(length=8):
    return "".join(
        random.choice(string.ascii_lowercase + string.digits)
        for _ in range(length))
