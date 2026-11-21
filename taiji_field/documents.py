"""Hexagram documents for the grounding encoder (Phase 1).

Each hexagram gets ONE document: its name plus a single-line modern-Chinese
reading authored for this project. The readings are deliberately plain and
short — they describe the *situation pattern* the hexagram stands for
(e.g. a deadlock, a fresh start, a burden too heavy to carry), so that a
sentence-embedding model can match a situation description to a hexagram
by resonance (cosine similarity).

No classical guaci (卦辞) text is included: the *Yijing* original text is
~3,000 years old and public domain, but every usable modern
edition/translation/annotation is copyrighted, and copying one from memory
risks reproducing a modern text. The one-line readings below are original
to this project (MIT, like the rest of the repo). If a clearly
public-domain classical source is vendored later, it should be added with
attribution and this note updated.

The document text is what gets embedded. Format per hexagram:
    "<name>：<one-line reading>"
"""

from __future__ import annotations

from taiji_field.data import load_hexagrams

# Original one-line modern-Chinese readings, authored for this project.
# Keyed by King Wen number. Keep each to one plain sentence about the
# situation pattern — no classical quotations, no invented history.
READINGS: dict[int, str] = {
    1: "创造与主导：势头向上，主动开创，适合带头做事。",
    2: "顺应与承载：顺势而为，厚德载物，以柔成事。",
    3: "万事开头难：初生艰难，宜守不宜攻，先站稳脚跟。",
    4: "启蒙与学习：不懂就问，像学生一样虚心求教。",
    5: "等待时机：时机未到，耐心等待，养精蓄锐。",
    6: "争执与冲突：起了争端，宜和解退让，不宜硬争到底。",
    7: "组织与带队：需要纪律和统帅，像带队伍打仗一样行动。",
    8: "亲近与结盟：互相依靠，团结起来才有力量。",
    9: "小有积蓄：小有收获，力量还不够，继续积累。",
    10: "谨慎前行：如履薄冰，小心做事，守规矩。",
    11: "通达顺畅：上下沟通，局面打开了，否极泰来。",
    12: "闭塞僵局：上下不通，卡住了，先守住再说。",
    13: "与人和同：志同道合的人聚在一起，公开合作。",
    14: "丰收在手：收获很大，宜分享，不宜独占。",
    15: "谦逊低调：功成不居，越低调越稳。",
    16: "愉悦与准备：心情松快，但别只顾高兴，要预做准备。",
    17: "顺势跟随：跟着对的人走，顺应变化。",
    18: "整治积弊：内部出了问题，要动手整治、拨乱反正。",
    19: "临近与督导：亲自到场，监督推动，机会在眼前。",
    20: "观察与展示：先看清楚再做决定；自己也在被别人看着。",
    21: "咬合突破：中间有梗阻，要下决心啃掉它。",
    22: "装饰与体面：外表光鲜，内容为本，别只做表面。",
    23: "剥落衰败：根基被一点点剥掉，宜守不宜进。",
    24: "回归与复苏：低谷反弹，重新开始，一阳来复。",
    25: "顺其自然：不妄动，不强求，按本分做事。",
    26: "蓄势待发：积蓄力量，等到时机一到就大干。",
    27: "滋养与口腹：靠什么养自己，言语和饮食都要节制。",
    28: "负担过重：担子太重，结构快撑不住，要减负。",
    29: "重重险阻：一坑接一坑，险中求生，保持冷静。",
    30: "光明与依附：像火一样明亮，依附于正道才能持久。",
    31: "感应与心动：两心相感，互相吸引，情感先行。",
    32: "持久与恒定：长久坚持，靠的是稳定的节奏。",
    33: "退避与抽身：时机不对，体面地退一步，保存实力。",
    34: "力量强盛：气势正旺，但别逞强，守住分寸。",
    35: "晋升与前进：太阳升起，向上发展，宜主动争取。",
    36: "光明受损：环境黑暗，韬光养晦，保全自己。",
    37: "家与分工：家里各有其位，内外分工，先齐家。",
    38: "背离与分歧：看法不同，各走各的，求同存异。",
    39: "寸步难行：进退两难，走不动了，找人帮忙。",
    40: "解开与释放：困局松动了，赶紧脱身，别恋战。",
    41: "有所减损：先舍后得，减掉多余的才能轻装上路。",
    42: "有所增益：彼此受益，宜往前走，利人利己。",
    43: "决断与清除：到了该拍板的时候，公开决断，清除隐患。",
    44: "不期而遇：意外相遇，机会突然出现，也要防备。",
    45: "聚集与汇合：人、资源聚到一起，趁势成事。",
    46: "步步上升：像树一样慢慢长，稳步上升。",
    47: "困顿受阻：被困住了，内心要稳住，等转机。",
    48: "滋养与根基：井水取之不竭，养好根本，别只顾眼前。",
    49: "变革与革新：旧的要改，变革要顺应人心、讲清楚。",
    50: "立新与鼎立：新局面立起来了，宜稳固、宜分享成果。",
    51: "震动与惊醒：突然的震动让人警醒，动起来反而是好事。",
    52: "止与静：该停就停，守住边界，别乱动。",
    53: "循序渐进：像大雁一样一步步来，稳扎稳打。",
    54: "不当的位置：位置不对、名分不正，宜谨慎，别强求。",
    55: "丰盛与盛极：正是最盛的时候，盛极要防衰。",
    56: "羁旅在外：在外漂泊，靠人帮衬，小心行事。",
    57: "顺从与渗透：像风一样无孔不入，柔顺而坚持。",
    58: "喜悦与交流：以言语取悦人，坦诚沟通，皆大欢喜。",
    59: "涣散与离心：人心散了，要重新凝聚，分散反而能解困。",
    60: "节制与分寸：凡事有度，知道什么时候停。",
    61: "诚信在内：心中有诚信，别人自然信你。",
    62: "小有过越：小事上可以过一点，大事仍要守规矩。",
    63: "已成定局：事情成了，但别松懈，守成比创业难。",
    64: "未成定局：事情还没成，仍在路上，保持希望。",
}


def hexagram_documents() -> list[dict]:
    """One document per hexagram, in King Wen order.

    Each document is ``{"king_wen", "bits", "name_zh", "text"}`` where
    ``text`` is the string that gets embedded.
    """
    docs = []
    for h in load_hexagrams():
        reading = READINGS[h.king_wen]
        docs.append(
            {
                "king_wen": h.king_wen,
                "bits": h.bits,
                "name_zh": h.name_zh,
                "text": f"{h.name_zh}：{reading}",
            }
        )
    return docs
