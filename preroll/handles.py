"""预告片预卷/后卷的帧与音频对齐工具。

数据模型
--------
本模块把每条素材片段表示为内联 dict::

    {
        "clip_id": str,
        "video_start": int,
        "video_end": int,
        "content_start": int,
        "content_end": int,
        "audio_start": int,
        "audio": list[int | float],
        "sample_rate": int,
        "fps": int,
    }

* 帧号是源素材中的 0 基帧号，视频区间是半开区间 ``[video_start, video_end)``；
  帧数 = ``video_end - video_start``。
* ``[content_start, content_end)`` 是真正的成片内容，必须落在视频区间内部；
  内容两侧多出来的帧就是 handle（预卷/后卷）。没有 handle 时内容边界等于
  视频边界。
* 音频样本号也是 0 基；``audio[i]`` 是源音频中的第
  ``audio_start + i`` 个样本。``audio_start`` 是当前音频缓冲区第一个样本的
  源样本号；音画同步时它应等于 ``video_start`` 对应的样本号。
* 所有帧到样本换算都使用 ``round(frames * sample_rate / fps)``：先乘后除、
  最后只取整一次，不允许逐帧 floor 后累加。所有对齐判定的容差为 1 个样本。
"""


def _frames_to_samples(frames, sample_rate, fps):
    """先乘后除、只取整一次的帧->样本换算（与内建 round 的偶数舍入一致）。

    用整数运算避免大数下的浮点误差；不允许逐帧 floor 后累加。
    """
    numerator = frames * sample_rate
    quotient, remainder = divmod(numerator, fps)
    twice = remainder * 2
    if twice > fps:
        return quotient + 1
    if twice < fps:
        return quotient
    # 恰好 .5：round-half-to-even，与 Python 内建 ``round`` 一致。
    return quotient + (quotient & 1)


def expand_clip(clip, preroll_frames, postroll_frames, bounds):
    """把片段的视频区间向前后扩展，并钳制在素材边界内。

    参数
    ----
    clip : dict
        片段，见模块文档。
    preroll_frames : int
        向前增加的帧数，必须 >= 0。
    postroll_frames : int
        向后增加的帧数，必须 >= 0。
    bounds : dict
        ``{"start": int, "end": int}``，源素材可用的半开帧区间
        ``[start, end)``。``start < end``。

    返回
    ----
    dict
        入参的浅拷贝，只替换 ``video_start`` / ``video_end``。帧号规则：

        * ``video_start = max(bounds["start"], 原 video_start - preroll_frames)``
        * ``video_end = min(bounds["end"], 原 video_end + postroll_frames)``

        因而后卷只可能加在出点（``video_end``）一侧，入点不会被后卷推后。

    契约
    ----
    * 扩展后的帧长度是 ``video_end - video_start``，在未触边时等于
      ``原帧长度 + preroll_frames + postroll_frames``；触边时被钳制到素材边界，
      实际长度可以更短。
    * 越界的预卷/后卷必须裁掉，绝不能产生素材外的帧号。
    * ``content_start`` / ``content_end`` / ``audio`` / ``audio_start`` 原样复制：
      本函数只改编辑边界，不合成缺失的音频。扩展造成的音画偏移由
      :func:`align_audio` 修正、由 :func:`validate_clip` 报告。
    * 不修改入参。负 handle、非法 bounds、扩展后空区间或反向区间抛
      :class:`ValueError`。
    """
    if preroll_frames < 0 or postroll_frames < 0:
        raise ValueError("preroll_frames 和 postroll_frames 必须 >= 0")
    if clip["video_start"] >= clip["video_end"]:
        raise ValueError("原片段的视频区间为空或反向")
    bound_start = bounds["start"]
    bound_end = bounds["end"]
    if bound_start >= bound_end:
        raise ValueError("bounds 必须满足 start < end")

    new_start = max(bound_start, clip["video_start"] - preroll_frames)
    # 后卷只加在出点一侧，绝不加到入点。
    new_end = min(bound_end, clip["video_end"] + postroll_frames)
    if new_start >= new_end:
        raise ValueError(
            f"扩展后区间为空或反向: [{new_start}, {new_end})"
        )

    out = dict(clip)
    out["video_start"] = new_start
    out["video_end"] = new_end
    return out


def audio_offsets(video_start, audio_start, sample_rate, fps):
    """计算视频起点与音频起点之间的样本偏移。

    参数
    ----
    video_start : int
        视频起点的源帧号（0 基）。
    audio_start : int
        音频起点的源样本号（0 基）。
    sample_rate : int
        每秒音频样本数，必须 > 0。
    fps : int
        每秒视频帧数，必须 > 0。

    返回
    ----
    int
        ``round(video_start * sample_rate / fps) - audio_start``。正数表示音频
        起点偏早（需要向后推迟这么多样本），负数表示音频偏晚，0 表示同步。
        判定是否同步时允许 1 个样本的误差。

    契约
    ----
    * 先做乘法 ``video_start * sample_rate``，再除以 ``fps``，最后只取整一次。
      不能把 ``fps`` 当除数去缩小帧号，也不能用
      ``video_start * (sample_rate // fps)`` 这类逐帧 floor 的写法；后者在
      44100 Hz / 24 fps 下长片累计会漂移约一帧。
    * ``video_start`` 和 ``audio_start`` 都按源时间轴的 0 基位置解释。
    * 返回整数；``sample_rate <= 0``、``fps <= 0`` 或 ``video_start < 0`` 抛
      :class:`ValueError`。
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate 必须 > 0")
    if fps <= 0:
        raise ValueError("fps 必须 > 0")
    if video_start < 0:
        raise ValueError("video_start 必须 >= 0")

    video_sample = _frames_to_samples(video_start, sample_rate, fps)
    return video_sample - audio_start


def trim_to_content(clip, content_len):
    """去掉多余 handle，把视频区间裁到恰好 ``content_len`` 帧内容。

    参数
    ----
    clip : dict
        片段，见模块文档。
    content_len : int
        需要的正片长度，单位是视频帧，必须 > 0 且不超过
        ``content_end - content_start``。

    返回
    ----
    dict
        入参的浅拷贝，输出视频区间固定为
        ``[content_start, content_start + content_len)``，并同步更新
        ``content_end``。裁剪后的帧长度严格等于 ``content_len``，不能把
        预卷/后卷再算进去。

    音频
    ----
    假定 ``clip["audio"]`` 覆盖裁剪前的整个视频区间，``audio[0]`` 对应
    ``video_start``。设::

        head_frames = content_start - video_start
        tail_frames = video_end - (content_start + content_len)

    则裁掉头部 ``round(head_frames * sample_rate / fps)`` 个样本和尾部
    ``round(tail_frames * sample_rate / fps)`` 个样本，输出音频长度等于新区间
    的换算长度。若原缓冲区不够长，用 0 补齐；若偏长，截断。``audio_start``
    增加头部被裁掉的样本数。

    契约
    ----
    * 内容长度守恒：``out["video_end"] - out["video_start"] == content_len``。
    * 头部的预卷和尾部的后卷都必须去掉；handle 不计入内容长度。
    * 不修改入参；``content_len`` 非法或内容标记不合法抛 :class:`ValueError`。
    """
    video_start = clip["video_start"]
    video_end = clip["video_end"]
    content_start = clip["content_start"]
    content_end = clip["content_end"]

    if content_len <= 0:
        raise ValueError("content_len 必须 > 0")
    if not (video_start <= content_start < content_end <= video_end):
        raise ValueError("内容标记不合法：内容必须落在视频区间内部")
    if content_len > content_end - content_start:
        raise ValueError("content_len 不能超过可用内容长度")

    sample_rate = clip["sample_rate"]
    fps = clip["fps"]
    new_video_start = content_start
    new_video_end = content_start + content_len

    head_frames = content_start - video_start
    # 尾部后卷的帧数（video_end - 新区间终点）；缓冲充足时切片终点
    # head_samples + target_samples 恰好把对应样本排除在外。
    _tail_frames = video_end - new_video_end
    head_samples = _frames_to_samples(head_frames, sample_rate, fps)
    target_samples = _frames_to_samples(content_len, sample_rate, fps)

    audio = clip["audio"]
    # 按 audio[0] 对应 video_start 的绝对位置切片：头部预卷落在切片之前被裁掉，
    # 尾部后卷落在切片终点之后被裁掉，handle 不计入内容。缓冲区不足时切片
    # 自然变短，再在尾部补 0；缓冲偏长时由最终截断收口。
    kept = list(audio[head_samples: head_samples + target_samples])
    if len(kept) >= target_samples:
        new_audio = kept[:target_samples]
    else:
        new_audio = kept + [0] * (target_samples - len(kept))

    out = dict(clip)
    out["video_start"] = new_video_start
    out["video_end"] = new_video_end
    out["content_end"] = new_video_end
    out["audio"] = new_audio
    out["audio_start"] = clip["audio_start"] + head_samples
    return out


def align_audio(clip, offset_samples):
    """把音频整体平移 ``offset_samples`` 个样本，并保持缓冲区长度不变。

    参数
    ----
    clip : dict
        片段，见模块文档。
    offset_samples : int
        有符号平移量。**正数表示把音频向后推迟**（在头部补静音、丢弃尾部），
        负数表示提前（丢弃头部、在尾部补静音），0 表示原地不动。

    返回
    ----
    dict
        入参的浅拷贝，``audio`` 长度与输入完全相同：

        * ``offset_samples >= 0``：``[0] * offset + audio[:-offset]``，再截到原长；
        * ``offset_samples < 0``：``audio[-offset:] + [0] * (-offset)``，再截到原长。

        同时 ``audio_start -= offset_samples``（头部样本对应的源位置随平移改变）。
        视频边界和内容边界不变。

    契约
    ----
    * 填充必须在头部、裁剪必须在尾部（正偏移时），否则声音会整体拖后。
    * 负偏移方向相反，但同样必须保持 ``len(out["audio"]) == len(clip["audio"])``。
    * 平移量绝对值超过缓冲区长度时，结果仍应是原长度的全零/有效截断，不能抛错，
      也不能改变长度。
    * 不修改入参。
    """
    audio = clip["audio"]
    length = len(audio)

    if offset_samples >= 0:
        # 正偏移：头部补静音、尾部丢弃，声音整体向后推迟。
        offset = min(offset_samples, length)
        new_audio = [0] * offset + list(audio[: length - offset])
    else:
        # 负偏移：丢弃头部、尾部补静音，声音整体提前。
        shift = min(-offset_samples, length)
        new_audio = list(audio[shift:]) + [0] * shift
    new_audio = new_audio[:length]

    out = dict(clip)
    out["audio"] = new_audio
    out["audio_start"] = clip["audio_start"] - offset_samples
    return out


def merge_handles(prev, nxt, max_gap):
    """合并两个相邻片段，重叠的 handle 只保留一次。

    参数
    ----
    prev, next : dict
        两个片段；``prev.video_start <= next.video_start``，且两者的
        ``sample_rate`` / ``fps`` 必须相同。
    max_gap : int
        允许的最大绝对间隔（帧）。设::

            gap = next["video_start"] - prev["video_end"]

        ``abs(gap) <= max_gap`` 时允许合并；``gap < 0`` 表示两个片段重叠。
        ``max_gap`` 必须 >= 0。

    返回
    ----
    dict
        合并后的片段：

        * ``video_start = prev.video_start``，
          ``video_end = max(prev.video_end, next.video_end)``；
        * 重叠帧只出现一次：从 ``next["audio"]`` 头部裁掉
          ``-gap`` 帧对应的样本后再拼接；
        * ``gap > 0`` 时在拼接处补 ``gap`` 帧的静音，保证合并区间连续；
        * 若 ``next`` 完全落在 ``prev`` 内，保留 ``prev`` 的音频，不重复追加；
        * ``audio_start = prev.audio_start``，``merged_ids = [prev_id, next_id]``，
          ``clip_id`` 为 ``f"{prev_id}+{next_id}"``。

    契约
    ----
    * 合并结果不能有重复播放的帧：两个片段首尾相接（``gap == 0``）时，合并后
      帧长度和音频样本数都等于两段之和。
    * 不能简单把两个 ``audio`` 列表首尾相接；那会把重叠段播两遍。
    * 不修改入参；间隔超过 ``max_gap``、顺序错误、采样率或帧率不一致抛
      :class:`ValueError`。
    """
    if max_gap < 0:
        raise ValueError("max_gap 必须 >= 0")
    if prev["video_start"] > nxt["video_start"]:
        raise ValueError("prev 必须位于 next 之前（video_start 有序）")
    if prev["sample_rate"] != nxt["sample_rate"] or prev["fps"] != nxt["fps"]:
        raise ValueError("两个片段的 sample_rate / fps 必须一致")

    gap = nxt["video_start"] - prev["video_end"]
    if abs(gap) > max_gap:
        raise ValueError(
            f"间隔 {gap} 帧超过 max_gap={max_gap}，不允许合并"
        )

    sample_rate = prev["sample_rate"]
    fps = prev["fps"]
    video_start = prev["video_start"]
    video_end = max(prev["video_end"], nxt["video_end"])

    prev_audio = list(prev["audio"])
    nxt_audio = list(nxt["audio"])

    if nxt["video_end"] <= prev["video_end"]:
        # next 完全落在 prev 内：保留 prev 的音频，不重复追加。
        merged_audio = prev_audio
    else:
        if gap < 0:
            # 重叠：从 next 头部裁掉重叠帧对应的样本，重叠段只播一次。
            overlap_samples = _frames_to_samples(-gap, sample_rate, fps)
            nxt_part = nxt_audio[overlap_samples:]
            merged_audio = prev_audio + nxt_part
        elif gap > 0:
            # 有间隔：拼接处补 gap 帧的静音，保证合并区间连续。
            silence_samples = _frames_to_samples(gap, sample_rate, fps)
            merged_audio = prev_audio + [0] * silence_samples + nxt_audio
        else:
            # 首尾相接：直接拼接，帧数与样本数都是两段之和。
            merged_audio = prev_audio + nxt_audio

    return {
        "clip_id": f"{prev['clip_id']}+{nxt['clip_id']}",
        "video_start": video_start,
        "video_end": video_end,
        "content_start": prev["content_start"],
        "content_end": nxt["content_end"],
        "audio_start": prev["audio_start"],
        "audio": merged_audio,
        "sample_rate": sample_rate,
        "fps": fps,
        "merged_ids": [prev["clip_id"], nxt["clip_id"]],
    }


def duration_samples(clip, sample_rate, fps):
    """把片段的视频帧长度换算成音频样本数。

    参数
    ----
    clip : dict
        片段，见模块文档。
    sample_rate : int
        每秒音频样本数，必须 > 0。
    fps : int
        每秒视频帧数，必须 > 0。

    返回
    ----
    int
        ``round((video_end - video_start) * sample_rate / fps)``，按最接近的
        整数取整。换算结果允许 1 个样本的误差。

    契约
    ----
    * 必须使用整段帧数先乘后除，不能 ``int(...)`` 截断，也不能逐帧截断后累加；
      长片段累计少几百个样本属于缺陷。
    * 帧长度必须 > 0；``sample_rate <= 0`` 或 ``fps <= 0`` 抛
      :class:`ValueError`。
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate 必须 > 0")
    if fps <= 0:
        raise ValueError("fps 必须 > 0")
    frame_count = clip["video_end"] - clip["video_start"]
    if frame_count <= 0:
        raise ValueError("片段帧长度必须 > 0")
    return _frames_to_samples(frame_count, sample_rate, fps)


def validate_clip(clip, bounds):
    """检查片段的边界、handle 符号和音画同步，返回问题代码列表。

    参数
    ----
    clip : dict
        片段，见模块文档。
    bounds : dict
        ``{"start": int, "end": int}``，源素材可用的半开帧区间。

    返回
    ----
    list[str]
        按下列固定顺序返回，空列表表示健康：

        * ``"out_of_bounds:{clip_id}"``：``video_start < bounds.start``、
          ``video_end > bounds.end``，或 ``video_start >= video_end``；
        * ``"negative_handle:{clip_id}"``：内容标记不合法，即
          ``content_start < video_start``、``content_end > video_end``，或
          ``content_start >= content_end``；
        * ``"audio_desync:{clip_id}"``：``audio_start`` 与 ``video_start`` 的
          样本偏移绝对值超过 1 个样本（容差 1 样本）。

    契约
    ----
    * 同一类问题对同一片段最多报一次，不同类问题可以同时出现。
    * 只报告、不修复、不修改入参；不因上述三类问题抛异常。
    * 音画同步必须实际比较 ``audio_start`` 与 ``video_start`` 换算后的样本号，
      不能只看两个字段是否相等。
    """
    clip_id = clip["clip_id"]
    problems = []

    video_start = clip["video_start"]
    video_end = clip["video_end"]
    if (
        video_start < bounds["start"]
        or video_end > bounds["end"]
        or video_start >= video_end
    ):
        problems.append(f"out_of_bounds:{clip_id}")

    content_start = clip["content_start"]
    content_end = clip["content_end"]
    if (
        content_start < video_start
        or content_end > video_end
        or content_start >= content_end
    ):
        problems.append(f"negative_handle:{clip_id}")

    # 实际换算视频起点对应的源样本号，与 audio_start 比较（容差 1 样本）。
    expected_audio = _frames_to_samples(
        video_start, clip["sample_rate"], clip["fps"]
    )
    if abs(expected_audio - clip["audio_start"]) > 1:
        problems.append(f"audio_desync:{clip_id}")

    return problems


def preroll_batch(clips, config):
    """批量套用预卷/后卷配置，并为每条片段独立计算音频偏移。

    参数
    ----
    clips : list[dict]
        片段列表，顺序即输出顺序。
    config : dict
        ``{"preroll_frames": int, "postroll_frames": int,
        "bounds": {"start": int, "end": int}}``；两个 handle 字段缺省为 0。

    返回
    ----
    dict
        ::

            {
                "clips": [
                    {
                        ...expand_clip 返回的全部字段...,
                        "audio_offset": int,
                    },
                    ...
                ],
                "count": int,
                "total_frames": int,
            }

        每条结果的 ``audio_offset`` 都用**该片段自己的** ``video_start``、
        ``audio_start``、``sample_rate``、``fps`` 计算：
        ``audio_offsets(expanded["video_start"], clip["audio_start"],
        clip["sample_rate"], clip["fps"])``。``total_frames`` 是所有扩展后
        片段的帧长度之和。

    契约
    ----
    * 各片段相互独立：不能复用上一片的 offset、缓冲区或临时变量，也不能让
      前一片的 bounds 钳制影响后一片。
    * 不修改 ``clips`` 或 ``config``；输入项的顺序必须保留。
    * 某条片段非法时抛 :class:`ValueError`，错误应指出 ``clip_id``。
    """
    preroll_frames = config.get("preroll_frames", 0)
    postroll_frames = config.get("postroll_frames", 0)
    clip_bounds = config["bounds"]

    result_clips = []
    total_frames = 0
    for clip in clips:
        try:
            expanded = expand_clip(
                clip, preroll_frames, postroll_frames, clip_bounds
            )
            # 每条片段用自己的起点与音频参数独立计算，禁止跨片段复用。
            expanded["audio_offset"] = audio_offsets(
                expanded["video_start"],
                clip["audio_start"],
                clip["sample_rate"],
                clip["fps"],
            )
        except ValueError as exc:
            raise ValueError(
                f"片段 {clip.get('clip_id')!r} 处理失败: {exc}"
            ) from exc
        total_frames += expanded["video_end"] - expanded["video_start"]
        result_clips.append(expanded)

    return {
        "clips": result_clips,
        "count": len(result_clips),
        "total_frames": total_frames,
    }
