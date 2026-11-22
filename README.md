# video-audio-preroll

## 题面

剪预告片时给每段留了预卷，结果声音总是抢在画面之前：加后卷时把后卷加到入点，片段反而短了一截；预卷扩展不查素材边界，起幅被推到素材外；音画起点换算把每秒帧数当除数，样本偏移直接翻倍；帧到样本逐帧向下取整，长片累计错开一整帧；按内容回裁把前后 handle 也计入内容，成片越裁越长；音频对齐把静音补在尾部，声音整体拖后；相邻片段合并 handle 时默许区间重叠，接缝处的音频播两遍；时长换样本用 int 截断，长片段少掉几百个样本；校验只看帧边界和 handle 符号，音画起点错位也不报；批量处理复用第一片的音频偏移，第二片起手就偏。preroll/handles.py 是核心，tests/test_handles.py 15个红。要修的是：…

## 运行

在仓库根目录执行：

```bash
python -m pytest tests -q
```

初始状态应恰好得到 `15 failed`，不应出现 collection、import 或 syntax error。测试只依赖 Python 标准库和 pytest，离线即可运行，不使用随机数，也不 sleep。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 测试不可修改

`tests/test_handles.py` 是验收契约，不可修改；只允许修改 `preroll/handles.py`。

### clip 字段

片段使用内联 dict，帧号是 0 基源帧号，视频区间为半开区间 `[video_start, video_end)`；音频样本号也是 0 基，`audio[0]` 对应源样本 `audio_start`：

```python
{
    "clip_id": str,
    "video_start": int,
    "video_end": int,
    "content_start": int,
    "content_end": int,
    "audio_start": int,
    "audio": list,          # int/float 样本，按播放顺序
    "sample_rate": int,     # 每秒音频样本数
    "fps": int,             # 每秒视频帧数
}
```

`[content_start, content_end)` 是正片内容，必须落在视频区间内；两侧多出来的帧是预卷/后卷 handle。所有帧到样本换算使用 `round(frames * sample_rate / fps)`，所有对齐判定容差为 1 个样本。
