# LLM 基线 v1（Baseline v1）

**日期：** 2026-11-21
**状态：⚠️ 未完成——调用被拦下，见下。没有基线数字。**

## 发生了什么

基线运行器已经写好（`scripts/run_baseline.py`，见"方法"），
离线部分全部验证通过（prompt 生成、JSON 解析、评分定义——见
`tests/test_measurement.py`）。但第一次真实调用（2 题冒烟测试）
没有跑通：

- 同一会话稍早，连接器自带的探针
  `python3 .../deepseek.py chat "Say ok."` 已获批并在约 6 秒内返回，
  说明 DeepSeek 连接本身是通的，模型 `deepseek-chat` 可用。
- 冒烟测试的第一次 API 调用在 120 秒后报 `urllib.error.URLError:
  timed out`，同时该次执行的审批卡片无人应答、超时作废。
- 按规则，未获批的部分不重试。所以 30 次调用一次也没跑成，
  `taiji_field/data/baseline_v1.json` **不存在**——没有数字可写，
  也不编造。

## 恢复办法（给下一步）

拿到审批后，直接跑：

```sh
python scripts/run_baseline.py
```

30 次调用，每次间隔 1 秒，温度 0。若中途断掉，用
`--resume` 续跑已记录的题，不重复花钱：

```sh
python scripts/run_baseline.py --resume
```

跑完后 `taiji_field/data/baseline_v1.json` 即为基线结果，
本文件要更新：填入 top-1 / path 数字，并补一节"与系统数字的比较"。

## 方法（已实现，未执行）

- **模型：** DeepSeek `deepseek-chat`，temperature 0，max_tokens 400。
  用 `custom.deepseek` 连接器自己的机制（`dynamic_credentials`
  替代凭证交换，`api.deepseek.com`），与连接器 helper `deepseek.py`
  同款做法。没有从任何地方提取 API key。
- **公平性：** 每题一次调用。Prompt = 情境文本 + 64 卦全部信息
  （卦名 + 一句现代解读，取自 `taiji_field/documents.py` 的
  `READINGS`）——恰好是接地编码器 v1 拿到的一样信息，不多不少：
  没有段落、没有上下卦分解、没有图关系、没有提议标签的提示。
  Prompt 全文以模板形式存在脚本里，运行结果文件会原样记录
  `prompt_template` 与 `hexagram_list`，可复现。
- **输出：** 要求 JSON（起始卦编号+卦名、动爻 1–3 个、落点卦编号、
  一句话理由）。落点卦由脚本从起始卦 + 动爻算出（`flip_bit`），
  不信任模型自己算的落点；模型算的落点另行记录以备对照。
  解析失败的回答重试一次，仍失败则记为 `parsed: false`、按 miss
  计分，绝不编造。
- **评分：** 与考试 harness（`taiji_field/exam.py`）完全相同的定义：
  top-1 = 预测起始卦等于提议起始卦；path = 起始卦与动爻都对
  （起始卦错了，动爻对了也不算——这是 Phase 2 笔记里记下的那个
  bug 的回归测试，见 `tests/test_measurement.py`）。

## 校准问题（基线跑完才能回答）

Roadmap 第 6 节的可证伪点是：状态动力学在卷子上赢不赢得过
token 预测。没有基线，这个问题目前无法回答。系统在 30 题上的
数字是 top-1 15/30（薄文档）/ 11/30（丰富文档）、path 2/30；
在 100 题上是 top-1 10/100、path 1/100（见
[docs/exam-100-notes.md](exam-100-notes.md)）。基线一到，
第一件事就是把同一张卷子上的两组数字并排放进这里。

## 诚实边界

1. **基线不存在。** 本文件目前只记录方法和一次失败的尝试，
   不是结果。任何引用"LLM 基线多少分"的说法在基线跑完之前
   都是错的。
2. **标签是草稿。** 30 题的提议演化是单人草稿、未审（见
   `docs/phase1-notes.md`）。基线与系统比的是"与草稿标签的
   一致程度"，不是"正确程度"。
3. **一次调用的 LLM 预测自带方差。** 温度 0 降低了随机性，
   但不同日期、不同模型版本仍可能给出不同答案；结果文件
   记录模型名与 prompt，复现时要核对。
