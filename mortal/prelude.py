import sys
import logging
import warnings
import torch
import numpy as np
from cpu_affinity import maybe_configure_process_affinity

sys.stdin.reconfigure(encoding='utf-8')

logging.basicConfig(
    stream = sys.stderr,
    level = logging.INFO,
    format = '%(asctime)s %(levelname)8s %(filename)12s:%(lineno)-4s %(message)s',
)

warnings.simplefilter('ignore')

# "The given NumPy array is not writeable"
dummy = np.array([])
dummy.setflags(write=False)
torch.as_tensor(dummy)

# "distutils Version classes are deprecated"
import torch.utils.tensorboard

warnings.simplefilter('default')

maybe_configure_process_affinity(context='main process')
