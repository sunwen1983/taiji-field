"""Build the Phase 1 exam paper from its scenario table.

Source of truth for the exam is the SCENARIOS table below. The end
hexagram of each proposed evolution is COMPUTED here (start hexagram +
moving lines, via taiji_field.graph.flip_bit) — never hand-typed — so a
proposed label can never be internally inconsistent.

Outputs (both generated, do not edit by hand):
- taiji_field/data/exam_v1.json  (machine-readable)
- docs/exam-v1.md                (human-readable)

Usage: python scripts/build_exam.py
"""

from __future__ import annotations

import json
from pathlib import Path

from taiji_field.data import load_hexagrams
from taiji_field.graph import flip_bit

ROOT = Path(__file__).resolve().parent.parent

# Each scenario: (id, category, scenario text, start King Wen, moving lines,
#                 one-line rationale for the proposed evolution).
# Categories: 工作决策 / 谈判 / 关系 / 创意 / 日常.
# Labels are DRAFTS proposed by the author — awaiting Wen's review,
# NOT ground truth (see docs/exam-v1.md).
SCENARIOS = [
    ("E01", "工作决策",
     "手里有个很好的新机会，薪水更高、平台更大，但现在的工作也还算稳定，团队也熟。要不要跳槽？",
     26, [6], "积累到顶，天路自开：动顶爻，跳槽后上下通达。"),
    ("E02", "工作决策",
     "项目到了关键节点，团队已经连续加班一个月，有人开始抱怨、效率也在掉。负责人该怎么办？",
     28, [1], "担子太重，到了该决断的时候：砍掉多余的任务，给团队减负。"),
    ("E03", "工作决策",
     "两个部门为一个项目的归属僵持了三周，谁也不让步，会议越开气氛越僵。",
     12, [5], "否到极处，高位的人先松动一点，局面就开始向上走。"),
    ("E04", "工作决策",
     "一个很看好的项目，立项申请已经交上去两个月了，上面迟迟没有批复。团队是继续等，还是先做点别的？",
     5, [6], "等久了别干等：先把手头的小事积累起来，等批复一到就能动。"),
    ("E05", "工作决策",
     "团队内部流程积弊已久：审批要跑五个人，出了问题互相推诿。有人提议彻底整顿，但怕动静太大。",
     18, [6], "整治要从上到下做到底：整顿完了，局面开始往上走。"),
    ("E06", "工作决策",
     "一个新人入职半年，进步很快，领导想提拔他，但团队里有资历更老的同事。怎么处理才稳妥？",
     46, [2], "稳步上升，但要谦逊低调：提拔要让人服气，别张扬。"),
    ("E07", "谈判",
     "和供应商谈续约，对方坚持涨价 20%，我们觉得最多接受 8%。谈了三轮，谁也不让步。",
     6, [4], "争执不下，不如先散开冷静：各自退一步，换个时间再谈。"),
    ("E08", "谈判",
     "公司和客户对交付标准理解不一致，客户认为我们没做到合同要求，我们认为已经超额完成。气氛很僵。",
     12, [1], "僵局中先别妄动：顺其自然，把事实摆清楚，局自己会松动。"),
    ("E09", "谈判",
     "两个合作方对产品方向意见完全相反：一方要做高端，一方要走量。中间卡着一个谁也绕不开的技术难题。",
     38, [2], "分歧中间有梗阻：先把那个技术难题啃掉，分歧才有谈的余地。"),
    ("E10", "谈判",
     "三家公司想联合做一个项目，大家目标一致，但谁出多少资源、谁说了算还没谈拢。",
     13, [3], "人多合作，别强求：顺其自然，先把能做的部分做起来。"),
    ("E11", "谈判",
     "和客户谈年度预算，对方想砍掉三分之一，但我们的成本摆在那里，砍了就做不下来。",
     60, [6], "守住分寸，谈到最后靠诚信：把账算清楚，对方自然信你。"),
    ("E12", "谈判",
     "竞争对手突然来挖我们的核心员工，开出的条件很好。员工还没决定，先来找我们谈。",
     44, [1], "意外相遇也是机会：顺势做大，拿出诚意把人留住、把事做大。"),
    ("E13", "关系",
     "暗恋一个人很久了，对方似乎也有好感，但谁也没捅破窗户纸。到底要不要表白？",
     31, [3], "两心相感，人聚到一起：表白了，关系就从感应变成汇合。"),
    ("E14", "关系",
     "家里家务分工一直不均，一个人做得多、怨气也多，另一个人觉得理所当然。怎么开口谈才不吵架？",
     37, [3], "家里分工调顺了，彼此都受益：先把分工摆到台面上谈清楚。"),
    ("E15", "关系",
     "和一个老朋友因为误会疏远了半年，其实双方都想修复，但谁也不好意思先开口。",
     8, [5], "亲近到一定程度，顺应承载：一方先伸手，关系自然回到从前。"),
    ("E16", "关系",
     "一个社群本来很活跃，最近人心散了，活动也没人来。组织者想重新凝聚大家。",
     59, [2], "人心散了，先看清楚局势：看明白大家到底想要什么，再凝聚。"),
    ("E17", "关系",
     "一次失信之后，合作伙伴对你的信任打了折扣，合作还在继续，但气氛明显不如从前。",
     61, [3], "信任修复后先小步积累：别一下求大，用一件件小事把信任挣回来。"),
    ("E18", "关系",
     "一段关系里总觉得自己位置不对：付出很多，名分却没有，越想越委屈。",
     54, [2], "位置不对，震动一下惊醒：重新定位，该谈的谈清楚。"),
    ("E19", "创意",
     "脑子里有个全新的产品点子，没人做过，自己很兴奋，但不知道从哪下手。",
     1, [1], "带头做事，引来意外相遇：先动起来，机会和同伴会在路上出现。"),
    ("E20", "创意",
     "一个作品折腾了半年，终于立起来了：结构完整，也有人愿意用。下一步该往哪使劲？",
     50, [1], "新局面立住了，开始有收获：先把成果巩固下来，再图扩大。"),
    ("E21", "创意",
     "作品内容不错，但包装很糙，别人第一眼没兴趣点进来。是花时间打磨包装，还是先堆内容？",
     22, [4], "装饰之下要见光明：包装要让人一眼看到内容在发光。"),
    ("E22", "创意",
     "一个想法很好，但推不动：开会讲了几次，大家听听就过去了。怎么让它渗透进去？",
     57, [2], "像风一样渗透，循序渐进：别指望一次说服，小步推进。"),
    ("E23", "创意",
     "团队刚拿了一个奖，大家都很兴奋，士气正高。趁热打铁开新项目，还是先冷静一下？",
     16, [1], "高兴过头要惊醒一下：先把准备工作做扎实，别光顾着乐。"),
    ("E24", "创意",
     "一个项目正处在最盛的时候：用户涨得快，口碑也好，但团队已经感到吃力了。",
     55, [4], "正盛时主动收敛、韬光养晦：盛极要防衰，别等到撑不住才停。"),
    ("E25", "日常",
     "深夜了，明天还有重要的事，是继续刷题还是去休息？手里的题刷到一半，停不下来。",
     29, [3], "一坑接一坑地刷，越刷越深：停下来，让井水重新蓄满。"),
    ("E26", "日常",
     "连续忙了两个月，身体发出信号：失眠、没胃口。是该彻底停下来休息一阵了。",
     52, [6], "该停就停，停到最后归于平和：彻底休息，养好再出发。"),
    ("E27", "日常",
     "经历了一段低谷：失业、失恋接连发生。最近感觉慢慢缓过来了，想重新开始。",
     24, [1], "刚复苏，别急着动：先顺应、承载，把脚跟站稳。"),
    ("E28", "日常",
     "总觉得生活像在透支：工作、社交、熬夜，样样都在消耗。想换个活法，先养好根本。",
     48, [1], "根基养好了，接下来是等时机：先把井养好，再谈取水。"),
    ("E29", "日常",
     "被一件烦心事困住了：想了很多办法都行不通，进也不是，退也不是。",
     47, [2], "被困时人聚到一起：找人商量，合力找出路。"),
    ("E30", "日常",
     "一件拖了很久的事，试了好几次都没成。心里还抱着希望，但不知道还要不要继续。",
     64, [6], "拖了很久的事终于松动、解开了：再推一把，困局就散了。"),
]


def build() -> dict:
    by_kw = {h.king_wen: h for h in load_hexagrams()}
    by_bits = {h.bits: h for h in load_hexagrams()}
    scenarios = []
    for sid, category, text, start_kw, lines, rationale in SCENARIOS:
        start = by_kw[start_kw]
        bits = start.bits
        for line in lines:
            bits = flip_bit(bits, line)
        end = by_bits[bits]
        scenarios.append(
            {
                "id": sid,
                "category": category,
                "scenario": text,
                "proposed": {
                    "start_king_wen": start.king_wen,
                    "start_name": start.name_zh,
                    "start_bits": start.bits,
                    "moving_lines": lines,
                    "end_king_wen": end.king_wen,
                    "end_name": end.name_zh,
                    "end_bits": end.bits,
                    "rationale": rationale,
                },
                "label_status": "draft — proposed by the author, awaiting Wen's review, not ground truth",
            }
        )
    return {
        "version": "v1",
        "note": "DRAFT labels. Each proposed evolution was proposed by the author and is awaiting Wen's review; it is NOT ground truth. End hexagrams are computed from start + moving lines, never hand-typed.",
        "scenarios": scenarios,
    }


def write_markdown(data: dict) -> str:
    lines = [
        "# 考试卷子 v1（Exam Paper v1）",
        "",
        "**状态：草稿标签（DRAFT）。** 每一题的“合理演化”都是作者提议的，**等待 Wen 审阅，不是标准答案**。Wen 审阅后，不一致的题要丢掉或讨论修改（roadmap 第 6 节：人工共识标注）。",
        "",
        "卷子结构：30 个小情境（工作决策 6、谈判 6、关系 6、创意 6、日常 6）。",
        "输入是一个情境（短文本）；输出是合理演化：起始卦 + 动爻 + 落点卦 + 一句为什么。",
        "落点卦由起始卦 + 动爻算出（`scripts/build_exam.py`），不是手填的。",
        "",
        "盲测基线（直接问 LLM“这个情境接下来会怎样”）是 Phase 1 计划的一部分，",
        "但本阶段未做：它需要 API key 和盲评流程，见 `docs/phase1-notes.md`。",
        "",
        "## 评分方式",
        "",
        "跑 `python -m taiji_field.exam`（或 `python scripts/run_exam.py`），",
        "对每题比较系统预测的 top-1 起始卦与提议的起始卦是否一致，",
        "并记录 top-3 是否命中、预测路径与提议路径是否一致。",
        "报告见 `taiji_field/data/exam_report.json`（由 harness 生成，不入库）。",
        "",
    ]
    current_category = None
    for s in data["scenarios"]:
        if s["category"] != current_category:
            current_category = s["category"]
            lines += ["", f"## {current_category}", ""]
        p = s["proposed"]
        lines += [
            f"### {s['id']} — {s['scenario'][:24]}……",
            "",
            s["scenario"],
            "",
            f"**提议演化（草稿）：** {p['start_name']} #{p['start_king_wen']}（{p['start_bits']}）"
            f" 动第 {'、'.join(str(x) for x in p['moving_lines'])} 爻 → "
            f"{p['end_name']} #{p['end_king_wen']}（{p['end_bits']}）。{p['rationale']}",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    data = build()
    assert len(data["scenarios"]) == 30
    json_path = ROOT / "taiji_field" / "data" / "exam_v1.json"
    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path = ROOT / "docs" / "exam-v1.md"
    md_path.write_text(write_markdown(data), encoding="utf-8")
    print(f"wrote {json_path} ({len(data['scenarios'])} scenarios)")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
