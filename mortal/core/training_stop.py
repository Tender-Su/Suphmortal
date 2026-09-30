"""Cooperative half of the supervisor's stop-file contract.

The supervisor owns wall-clock enforcement and hard termination. A stop-file
request is checked only where the trainer can safely checkpoint its gradients.
"""
import os
from pathlib import Path


ONLINE_STOP_REQUEST_EXIT_CODE = 87


def training_stop_requested(environ=None):
    environ = os.environ if environ is None else environ
    stop_file = environ.get('MORTAL_STOP_FILE', '')
    return bool(stop_file) and Path(stop_file).exists()
