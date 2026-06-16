"""Wrapper that patches torchaudio.load to use librosa on Windows."""
import sys
import numpy as np
import torch
import torchaudio
import librosa


def _patched_load(uri, frame_offset=0, num_frames=-1, normalize=True,
                  channels_first=True, format=None, buffer_size=4096, backend=None):
    offset = frame_offset / 44100 if frame_offset > 0 else 0.0
    duration = num_frames / 44100 if num_frames > 0 else None

    y, sr = librosa.load(str(uri), sr=None, mono=False, offset=offset, duration=duration)

    if y.ndim == 1:
        y = y[np.newaxis, :]  # mono -> (1, samples)

    waveform = torch.from_numpy(y.copy())
    if not channels_first:
        waveform = waveform.T

    return waveform, sr


torchaudio.load = _patched_load

from demucs.__main__ import main
sys.argv[1:1] = ["-n", "mdx"]
sys.exit(main())
