"""video-audio-preroll：预告片预卷、后卷与音画样本对齐。"""

from .handles import (
    align_audio,
    audio_offsets,
    duration_samples,
    expand_clip,
    merge_handles,
    preroll_batch,
    trim_to_content,
    validate_clip,
)

__all__ = [
    "expand_clip",
    "audio_offsets",
    "trim_to_content",
    "align_audio",
    "merge_handles",
    "duration_samples",
    "validate_clip",
    "preroll_batch",
]
