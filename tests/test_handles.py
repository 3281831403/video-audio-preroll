"""preroll.handles 的验收测试：15 个用例，全部调用 preroll/handles.py 的 8 个桩。

约定与 preroll/handles.py 的 docstring 一致：

* 帧号是 0 基源帧号，视频区间是半开区间 ``[video_start, video_end)``；
* 音频样本号也是 0 基，``audio[0]`` 对应源样本 ``audio_start``；
* 帧到样本换算为 ``round(frames * sample_rate / fps)``，容差 1 个样本；
* 所有用例离线、无随机数、不 sleep。

前 10 个用例分别锁定 10 个缺陷，后 5 个用例检查长度守恒、样本数、
合并时长、边界钳制和批量隔离。
"""

import pytest

from preroll.handles import (
    align_audio,
    audio_offsets,
    duration_samples,
    expand_clip,
    merge_handles,
    preroll_batch,
    trim_to_content,
    validate_clip,
)


def make_clip(
    clip_id="c",
    start=0,
    end=100,
    content_start=None,
    content_end=None,
    audio_start=0,
    audio=None,
    sample_rate=48000,
    fps=24,
):
    """构造一个符合模块文档的内联片段。"""
    if content_start is None:
        content_start = start
    if content_end is None:
        content_end = end
    return {
        "clip_id": clip_id,
        "video_start": start,
        "video_end": end,
        "content_start": content_start,
        "content_end": content_end,
        "audio_start": audio_start,
        "audio": [] if audio is None else audio,
        "sample_rate": sample_rate,
        "fps": fps,
    }


def bounds(start, end):
    return {"start": start, "end": end}


# ---------------------------------------------------------------- 缺陷 1：后卷加到入点
def test_expand_clip_puts_postroll_at_out_point():
    source = make_clip(
        "post", start=100, end=160, content_start=110, content_end=150
    )

    out = expand_clip(source, preroll_frames=10, postroll_frames=20, bounds=bounds(0, 1000))

    # 入点只减预卷，后卷只能加在出点。
    assert out["video_start"] == 90
    assert out["video_end"] == 180
    assert out["video_end"] - out["video_start"] == 90  # 60 + 10 + 20
    assert source["video_start"] == 100
    assert source["video_end"] == 160


# ---------------------------------------------------------------- 缺陷 2：预卷越出素材边界
def test_expand_clip_does_not_pull_preroll_past_source_start():
    source = make_clip(
        "edge", start=30, end=100, content_start=35, content_end=90
    )

    out = expand_clip(source, preroll_frames=25, postroll_frames=0, bounds=bounds(20, 200))

    assert out["video_start"] == 20
    assert out["video_start"] >= 20
    assert out["content_start"] == 35
    assert out["video_end"] == 100


# ---------------------------------------------------------------- 缺陷 3：用 fps 做除法
def test_audio_offsets_multiplies_frames_by_samples_per_frame():
    # 24 fps / 48000 Hz：每帧 2000 样本。
    assert audio_offsets(video_start=1, audio_start=2000, sample_rate=48000, fps=24) == 0
    assert audio_offsets(video_start=49, audio_start=98000, sample_rate=48000, fps=24) == 0

    # 49 帧对应 98000 样本；音频只有 96000 样本时偏早 2000 样本。
    assert audio_offsets(video_start=49, audio_start=96000, sample_rate=48000, fps=24) == 2000
    assert audio_offsets(video_start=49, audio_start=100000, sample_rate=48000, fps=24) == -2000


# ---------------------------------------------------------------- 缺陷 4：逐帧 floor 漂移
def test_audio_offsets_rounds_once_without_floor_drift():
    # 44100 / 24 = 1837.5 样本/帧；3601 帧一次取整应落在 6616838。
    expected = 6616838
    assert round(3601 * 44100 / 24) == expected
    assert audio_offsets(3601, expected, 44100, 24) == 0

    # 若写成 3601 * (44100 // 24)，会少 1801 个样本，约一帧。
    per_frame_floor = 3601 * (44100 // 24)
    assert per_frame_floor == 6615037
    assert expected - per_frame_floor == 1801


# ---------------------------------------------------------------- 缺陷 5：handle 计入内容长度
def test_trim_to_content_does_not_count_handles_as_content():
    # 14 帧素材：头 handle 2 帧、内容 10 帧、尾 handle 2 帧。
    source = make_clip(
        "trim",
        start=8,
        end=22,
        content_start=10,
        content_end=20,
        audio_start=8 * 2000,
        audio=list(range(14 * 2000)),
    )

    out = trim_to_content(source, content_len=10)

    assert (out["video_start"], out["video_end"]) == (10, 20)
    assert out["video_end"] - out["video_start"] == 10
    assert (out["content_start"], out["content_end"]) == (10, 20)
    assert out["audio_start"] == 10 * 2000
    assert out["audio"] == list(range(2 * 2000, 12 * 2000))
    assert len(out["audio"]) == 10 * 2000


# ---------------------------------------------------------------- 缺陷 6：静音补在尾部
def test_align_audio_pads_at_head_and_trims_tail():
    source = make_clip(
        "align",
        start=0,
        end=5,
        content_start=0,
        content_end=5,
        audio_start=1000,
        audio=[1, 2, 3, 4, 5],
        sample_rate=1,
        fps=1,
    )

    delayed = align_audio(source, offset_samples=2)
    assert delayed["audio"] == [0, 0, 1, 2, 3]
    assert delayed["audio_start"] == 998

    advanced = align_audio(source, offset_samples=-2)
    assert advanced["audio"] == [3, 4, 5, 0, 0]
    assert advanced["audio_start"] == 1002


# ---------------------------------------------------------------- 缺陷 7：合并时重叠播放
def test_merge_handles_plays_overlapping_audio_once():
    # 两段各 24 帧，重叠 4 帧（8000 样本）。
    prev = make_clip(
        "prev",
        start=0,
        end=24,
        content_start=4,
        content_end=20,
        audio_start=0,
        audio=list(range(24 * 2000)),
    )
    nxt = make_clip(
        "next",
        start=20,
        end=44,
        content_start=24,
        content_end=40,
        audio_start=20 * 2000,
        audio=list(range(100000, 100000 + 24 * 2000)),
    )

    out = merge_handles(prev, nxt, max_gap=4)

    assert (out["video_start"], out["video_end"]) == (0, 44)
    assert len(out["audio"]) == 44 * 2000  # 24 + 24 - 4 帧，只播一次
    assert out["audio"][: 24 * 2000] == prev["audio"]
    assert out["audio"][24 * 2000] == 108000  # next 头部被裁掉 8000 样本
    assert out["audio"][-1] == 147999
    assert 100000 not in out["audio"]


# ---------------------------------------------------------------- 缺陷 8：int 截断时长
def test_duration_samples_rounds_instead_of_truncating():
    one_frame = make_clip("one", start=0, end=1)
    assert duration_samples(one_frame, 44100, 24) == 1838  # 1837.5，四舍五入

    long_clip = make_clip("long", start=0, end=3601)
    assert duration_samples(long_clip, 44100, 24) == 6616838
    assert duration_samples(long_clip, 44100, 24) != int(3601 * 44100 / 24)


# ---------------------------------------------------------------- 缺陷 9：不校验音画同步
def test_validate_clip_reports_audio_desync_beyond_one_sample():
    valid = make_clip(
        "ok",
        start=100,
        end=200,
        content_start=110,
        content_end=190,
        audio_start=200000,  # 100 帧 * 2000
    )
    assert validate_clip(valid, bounds(0, 300)) == []

    # 1 个样本以内属于容差，不报。
    almost = make_clip(
        "almost",
        start=100,
        end=200,
        content_start=110,
        content_end=190,
        audio_start=200001,
    )
    assert validate_clip(almost, bounds(0, 300)) == []

    late = make_clip(
        "late",
        start=100,
        end=200,
        content_start=110,
        content_end=190,
        audio_start=202000,  # 音频晚 2000 样本
    )
    assert validate_clip(late, bounds(0, 300)) == ["audio_desync:late"]

    out_of_bounds = make_clip(
        "oob",
        start=140,
        end=220,
        content_start=150,
        content_end=210,
        audio_start=280000,
    )
    assert validate_clip(out_of_bounds, bounds(0, 200)) == ["out_of_bounds:oob"]

    negative_handle = make_clip(
        "neg",
        start=100,
        end=200,
        content_start=90,
        content_end=190,
        audio_start=200000,
    )
    assert validate_clip(negative_handle, bounds(0, 300)) == ["negative_handle:neg"]


# ---------------------------------------------------------------- 缺陷 10：批量复用 offset
def test_preroll_batch_computes_each_clip_offset_independently():
    first = make_clip(
        "first", start=0, end=240, content_start=0, content_end=240, audio_start=0
    )
    second = make_clip(
        "second",
        start=24,
        end=264,
        content_start=24,
        content_end=264,
        audio_start=49000,  # 视频 24 帧应对应 48000，音频偏晚 1000
    )
    config = {
        "preroll_frames": 0,
        "postroll_frames": 0,
        "bounds": bounds(0, 1000),
    }

    out = preroll_batch([first, second], config)

    assert out["count"] == 2
    assert out["total_frames"] == 480
    assert [item["audio_offset"] for item in out["clips"]] == [0, -1000]
    assert [item["clip_id"] for item in out["clips"]] == ["first", "second"]


# ---------------------------------------------------------------- 长度守恒：原长 + 前后卷
def test_expand_clip_length_is_original_plus_both_handles():
    source = make_clip(
        "length", start=100, end=200, content_start=120, content_end=180
    )

    for preroll, postroll in ((0, 0), (3, 7), (12, 5)):
        out = expand_clip(source, preroll, postroll, bounds(0, 1000))
        assert out["video_end"] - out["video_start"] == 100 + preroll + postroll
        assert out["content_start"] == 120
        assert out["content_end"] == 180
        assert source["video_start"] == 100
        assert source["video_end"] == 200


# ---------------------------------------------------------------- 样本数 = 帧数 × 每帧样本
def test_audio_sample_count_matches_video_frame_count():
    frames = 120
    source = make_clip(
        "count",
        start=24,
        end=24 + frames,
        content_start=24,
        content_end=24 + frames,
        audio_start=24 * 2000,
        audio=[0] * (frames * 2000),
    )

    sample_count = duration_samples(source, 48000, 24)

    assert sample_count == pytest.approx(frames * 2000, abs=1)
    assert sample_count == 240000
    assert len(source["audio"]) == sample_count


# ---------------------------------------------------------------- 相邻合并不重叠
def test_merge_handles_adjacent_duration_equals_sum():
    prev = make_clip(
        "a",
        start=0,
        end=24,
        content_start=0,
        content_end=24,
        audio_start=0,
        audio=list(range(24 * 2000)),
    )
    nxt = make_clip(
        "b",
        start=24,
        end=48,
        content_start=24,
        content_end=48,
        audio_start=24 * 2000,
        audio=list(range(100000, 100000 + 24 * 2000)),
    )

    out = merge_handles(prev, nxt, max_gap=0)

    assert (out["video_start"], out["video_end"]) == (0, 48)
    assert out["video_end"] - out["video_start"] == (24 - 0) + (48 - 24)
    assert len(out["audio"]) == 96000
    assert out["audio"][24 * 2000] == 100000
    assert out["audio"][-1] == 147999


# ---------------------------------------------------------------- 越界扩展被钳制到素材边界
def test_expand_clip_clamps_postroll_to_source_end():
    source = make_clip(
        "tail", start=80, end=100, content_start=82, content_end=98
    )

    out = expand_clip(source, preroll_frames=0, postroll_frames=30, bounds=bounds(0, 110))

    assert (out["video_start"], out["video_end"]) == (80, 110)
    assert out["video_end"] - out["video_start"] == 30
    assert out["video_end"] - out["video_start"] == (100 - 80) + 10  # 只吃下 10 帧后卷
    assert out["video_end"] <= 110


# ---------------------------------------------------------------- 批量与单片结果一致
def test_preroll_batch_matches_solo_processing_and_keeps_inputs():
    first = make_clip(
        "batch-a", start=0, end=240, content_start=20, content_end=220, audio_start=0
    )
    second = make_clip(
        "batch-b",
        start=240,
        end=480,
        content_start=260,
        content_end=460,
        audio_start=480000,
    )
    clips = [first, second]
    config = {
        "preroll_frames": 10,
        "postroll_frames": 10,
        "bounds": bounds(0, 600),
    }
    clips_snapshot = [dict(item) for item in clips]
    config_snapshot = {
        "preroll_frames": config["preroll_frames"],
        "postroll_frames": config["postroll_frames"],
        "bounds": dict(config["bounds"]),
    }

    batch = preroll_batch(clips, config)
    offsets = [item["audio_offset"] for item in batch["clips"]]
    assert offsets[0] != offsets[1]  # 两片起点不同，偏移不能复用

    for index, item in enumerate(clips):
        solo = preroll_batch([item], config)["clips"][0]
        assert batch["clips"][index] == solo

    assert clips == clips_snapshot
    assert config == config_snapshot
