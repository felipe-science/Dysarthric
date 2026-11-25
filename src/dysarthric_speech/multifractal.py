"""Implement the historical multifractal descriptor on decoded waveforms."""

import math

import numpy as np
from numpy.typing import NDArray
from scipy.stats import linregress

_ORDERS = np.linspace(-5, 5, 11)
_SCALES = (32, 64, 128, 256, 512, 1024, 2048, 4096)


def _detrended_rms(waveform: NDArray[np.floating], scale: int) -> NDArray[np.float64]:
    moving_average = np.zeros(len(waveform))
    for index in range(len(waveform)):
        # The historical causal window is truncated at the leading edge.
        start = max(0, index - scale + 1)
        moving_average[index] = np.mean(waveform[start : index + 1])

    residual = waveform - moving_average
    # The segment count excludes the last full block as in the experiment.
    block_count = math.floor(len(residual) / scale - 1)
    return np.array(
        [
            np.sqrt(np.mean(residual[start : start + scale] ** 2))
            for start in range(0, block_count * scale, scale)
        ]
    )


def _fluctuation(rms: NDArray[np.float64], order: np.float64) -> np.float64:
    if order == 0:
        return np.exp(np.mean(np.log(rms)))
    return np.mean(rms**order) ** (1 / order)


def delta_alpha(waveform: NDArray[np.floating]) -> float:
    """Return alpha at q=-5 minus alpha at q=5.

    The historical calculation can return NaN for short or silent recordings.
    """
    fluctuation = np.empty((len(_SCALES), len(_ORDERS)))
    for index, scale in enumerate(_SCALES):
        rms = _detrended_rms(waveform, scale)
        fluctuation[index] = [_fluctuation(rms, order) for order in _ORDERS]

    log_scale = np.log(_SCALES)
    slopes = np.array([linregress(log_scale, np.log(column)).slope for column in fluctuation.T])
    tau = slopes * _ORDERS - 1
    alpha = np.gradient(tau, _ORDERS)
    return float(alpha[0] - alpha[-1])
