"""Build the 100-scenario exam paper (exam_100.json) with multi-pass labels.

Source of truth is the SCENARIOS table below (pass A). The end hexagram
of each proposed evolution is COMPUTED here (start hexagram + moving
lines, via taiji_field.graph.flip_bit) — never hand-typed — so a
proposed label can never be internally inconsistent.

Annotation (see docs/exam-100-notes.md for the honest limits):

- Pass A: the author reasons from hexagram meanings (start hexagram +
  moving lines + one-line rationale), as in build_exam.py.
- Pass B: a DIFFERENT method — embedding-resonance retrieval (the
  grounding encoder's top-1 hexagram) + the v1 reasoner's path from
  that start — followed by a manual check of the disagreements (the
  author reads every A/B disagreement and judges whether B is a
  plausible alternative reading). Pass B never overrides pass A.
- Pass C: PENDING HUMAN REVIEW. No second LLM is available through
  the connectors (the Meta Model API connector is FixLens photo
  analysis only, not a general chat LLM), and the baseline model
  (DeepSeek deepseek-chat) must not also be the annotator, or the
  comparison would be biased.

Where A and B agree, the final label is that agreement. Where they
disagree, there is no majority (C is pending), so the tie-break takes
pass A and the scenario is flagged for human review.

Outputs (generated, do not edit by hand):
- taiji_field/data/exam_100.json

Usage: python scripts/build_exam_100.py
"""

from __future__ import annotations

import json
from pathlib import Path

from taiji_field.data import load_hexagrams
from taiji_field.exam import load_exam
from taiji_field.graph import flip_bit
from taiji_field.grounding import GroundingEncoder
from taiji_field.reasoning import reason

ROOT = Path(__file__).resolve().parent.parent

CATEGORIES = ["工作决策", "谈判", "关系", "创意", "日常"]

# Each scenario: (id, category, scenario text, start King Wen, moving lines,
#                 one-line rationale for the proposed evolution).
# Pass A labels are DRAFTS proposed by the author — NOT ground truth.
# 20 scenarios per category, 100 total.
SCENARIOS = [
    # ---- 工作决策 (F01–F20) ----
    ("F01", "工作决策",
     "公司要砍掉一个做了三年的老产品线，团队里有人反对、有人觉得早该砍了，负责人迟迟下不了决心。",
     43, [5], "决断到了中段：公开拍板、清除隐患，拖下去只会更难。"),
    ("F02", "工作决策",
     "新项目招不到合适的人，猎头推的人都不合适，是降低标准先招进来，还是继续等？",
     5, [4], "等待中别干等：先把手头能做的事做好，等合适的人出现。"),
    ("F03", "工作决策",
     "两个团队合并，职级和汇报线要重新排，有人担心自己被边缘化，气氛紧张。",
     8, [3], "结盟要公开讲清楚：含糊其辞只会让人猜疑。"),
    ("F04", "工作决策",
     "老板突然让接手一个烂尾项目，前任留下一堆坑，接还是推？",
     18, [1], "烂摊子要从最底层的问题动手整治。"),
    ("F05", "工作决策",
     "季度目标定得太高，团队拼尽全力也够不着，是申请下调目标，还是硬扛？",
     28, [2], "担子太重：先把目标调到结构撑得住的水平。"),
    ("F06", "工作决策",
     "核心员工提离职，手里还攥着关键客户，是加薪留人，还是放手？",
     45, [1], "人聚到一起：先把诚意摆出来，留人的事再谈条件。"),
    ("F07", "工作决策",
     "跨部门合作，对方总是不回消息、拖进度，是直接找对方老板，还是再忍忍？",
     6, [2, 5], "争执不下：先退一步找中间人，再由高位的人出面松动局面。"),
    ("F08", "工作决策",
     "公司推行新工具，团队抵触，说老办法挺好的，怎么推？",
     49, [3], "变革要讲清楚好处：时机未到硬推只会激起抵触。"),
    ("F09", "工作决策",
     "项目做了一半，发现当初的技术选型错了，重构代价很大，继续凑合还是推倒重来？",
     23, [5], "根基被剥掉大半：先保住能用的部分，再谈重构。"),
    ("F10", "工作决策",
     "年终奖分配，同事觉得不公平，私下抱怨，是公开标准还是私下安抚？",
     60, [4], "分寸要公开：标准摆清楚了，抱怨才有处安放。"),
    ("F11", "工作决策",
     "新业务方向，老板想 all in，团队觉得风险太大，怎么劝？",
     34, [4], "气势正旺：提醒守住分寸，强极易折。"),
    ("F12", "工作决策",
     "供应商突然断供，生产线要停，是紧急找替代，还是停下来等？",
     29, [2, 6], "险中求生：先找能顶上的替代方案，再把供应链的后路想清楚。"),
    ("F13", "工作决策",
     "团队里两个人因为分工吵起来了，谁也不服谁，负责人怎么调解？",
     38, [4], "各走各的：先找到两人都能接受的小共识。"),
    ("F14", "工作决策",
     "准备跳槽去创业公司，薪水涨 40% 但不稳定，留在大厂安稳但天花板明显，怎么选？",
     3, [1], "开头难：先想清楚自己能不能扛住最初的难，再动。"),
    ("F15", "工作决策",
     "公司要裁员，名单里有跟了自己三年的老员工，怎么开口谈？",
     41, [3], "有所减损：把补偿和出路谈清楚，减掉的才不至于伤人太重。"),
    ("F16", "工作决策",
     "客户要求的功能根本做不出来，销售已经答应了，怎么跟客户解释？",
     61, [4], "靠诚信：把做不到的实情讲清楚，对方反而可能信你。"),
    ("F17", "工作决策",
     "两个项目同时到 deadline，人手不够，保哪个、放哪个？",
     52, [5], "该停就停：保住一个、停掉一个，比两个都做砸强。"),
    ("F18", "工作决策",
     "入职新公司三个月，发现文化和预期差很多，是适应还是再找？",
     56, [2], "在外漂泊：先靠身边的人摸清情况，别急着下结论。"),
    ("F19", "工作决策",
     "团队连续拿了两个季度第一，老板想趁热打铁再定个更高的目标，团队已经很累了。",
     55, [4], "正盛时主动收敛：盛极要防衰，别等到撑不住才停。"),
    ("F20", "工作决策",
     "老客户突然要换供应商，理由是价格，是降价挽留还是放手？",
     42, [5], "利人利己：把长期合作的账算清楚，价格之外还有价值。"),
    # ---- 谈判 (F21–F40) ----
    ("F21", "谈判",
     "房东突然要涨租 30%，合同还没到期，是硬刚还是谈？",
     6, [2, 4], "争执：先把合同和依据摆出来，再退一步换个谈法，硬争到底两败俱伤。"),
    ("F22", "谈判",
     "和合伙人谈股份，对方要 60%，你觉得五五开才合理，谈了两次没谈拢。",
     13, [2], "志同道合：股份的事先放一放，把合作先跑起来再谈。"),
    ("F23", "谈判",
     "买二手房，中介说房东就这个价、不谈了，房子你很喜欢但超预算。",
     60, [5], "守住分寸：把预算底线亮清楚，谈不拢就走，别被喜欢冲昏头。"),
    ("F24", "谈判",
     "和投资人谈估值，对方压得很低，但公司账上钱只够三个月。",
     47, [1], "被困住了：先稳住，别在缺钱的时候贱卖公司。"),
    ("F25", "谈判",
     "和客户谈账期，对方要 90 天，我们只能接受 30 天，订单很大不想丢。",
     9, [3], "小有收获：先让一小步把订单保住，再慢慢谈账期。"),
    ("F26", "谈判",
     "公司并购谈判，对方要求核心团队签三年竞业，团队有人不愿意。",
     37, [2], "各有其位：把竞业的条件和补偿谈清楚，家里先达成一致。"),
    ("F27", "谈判",
     "和政府部门谈补贴，材料交了三轮，每次都说缺这个缺那个。",
     39, [3], "走不动了：找个熟悉流程的人带路，别自己硬闯。"),
    ("F28", "谈判",
     "离婚谈判，财产分割谈不拢，双方都请了律师，孩子夹在中间。",
     38, [3, 5], "背离：先把孩子的安排谈妥，再由高位的人让一步，财产的事慢慢谈。"),
    ("F29", "谈判",
     "和明星主播谈合作，对方要价高得离谱，但流量确实大。",
     14, [3], "收获很大：把分成谈成利益共享，别一次性买断。"),
    ("F30", "谈判",
     "和房东谈续租，想长租三年锁定租金，房东只想一年一签。",
     32, [2], "长久坚持：把三年里双方的收益算清楚，长约对房东也有好处。"),
    ("F31", "谈判",
     "供应商交货延期，合同里有违约金条款，是索赔还是先保供货？",
     21, [4], "中间有梗阻：先把货催到手，索赔的事等供货稳了再说。"),
    ("F32", "谈判",
     "和对手公司谈合并，两边老板都想当老大，谈判卡在控制权上。",
     7, [5], "需要纪律和统帅：控制权先定下来，否则合并了也是一盘散沙。"),
    ("F33", "谈判",
     "租办公室，物业要求押三付三，现金流吃紧，想谈成押一付三。",
     48, [3], "养好根本：把现金流的账算给对方看，押金是死的、合作是长的。"),
    ("F34", "谈判",
     "和老东家谈离职补偿，HR 说按最低标准，你想争取多一个月。",
     43, [3], "到了该拍板的时候：把诉求摆到台面上，别私下磨。"),
    ("F35", "谈判",
     "跨境合作，对方坚持用他们的合同模板，条款对我们不利。",
     56, [4], "在外漂泊：条款要一条一条抠，别图省事签了不利的约。"),
    ("F36", "谈判",
     "和经销商谈独家代理，对方要独家但不承诺销量。",
     8, [2], "互相依靠：独家可以给，但销量要对赌，结盟不能单方面付出。"),
    ("F37", "谈判",
     "拆迁谈判，邻居都签了，就剩你一家，开发商天天上门。",
     33, [1], "时机不对：先别硬顶，把条件谈到满意再退。"),
    ("F38", "谈判",
     "和学校谈合作办学，对方要冠名权，你方想保留品牌独立性。",
     22, [2], "外表光鲜：冠名可以谈，但内容和品牌底线要守住。"),
    ("F39", "谈判",
     "劳务纠纷谈判，员工索赔金额超出预期，公司想和解不想上法庭。",
     40, [2], "困局松动了：和解的窗口在，赶紧谈，别拖到对簿公堂。"),
    ("F40", "谈判",
     "和投资人谈对赌，业绩达不到要回购股份，创始人觉得风险太大。",
     28, [5], "担子太重：对赌条款要减到结构撑得住的水平。"),
    # ---- 关系 (F41–F60) ----
    ("F41", "关系",
     "父母催婚催得紧，每次回家都要吵一架，是顺从还是坚持自己的节奏？",
     37, [3], "家里各有其位：把自己的打算摆到台面上谈清楚。"),
    ("F42", "关系",
     "最好的朋友借了钱一直不还，开口要怕伤感情，不开口自己难受。",
     61, [3], "信任要靠一件件小事挣回来：先提个小数目试探。"),
    ("F43", "关系",
     "夫妻两人一个想留在大城市，一个想回老家，僵持半年了。",
     31, [2, 4], "两心相感：先别争地点，谈谈各自到底在怕什么，再把心意摆到台面上。"),
    ("F44", "关系",
     "孩子沉迷游戏，成绩下滑，管严了吵架、不管又担心。",
     4, [2], "不懂就问：先弄明白孩子在游戏里得到了什么，再谈管教。"),
    ("F45", "关系",
     "同事总在背后说你坏话，传到你耳朵里了，是当面对质还是装不知道？",
     36, [2], "环境黑暗：先保全自己，别正面硬碰。"),
    ("F46", "关系",
     "和父母住在一起，老人总插手带孩子的方式，矛盾越来越多。",
     37, [4], "各有其位：把带孩子的分工划清楚，谁负责谁说了算。"),
    ("F47", "关系",
     "暗恋的人突然主动找你聊天，聊得很投机，是趁热表白还是再等等？",
     31, [3], "两心相感：聊得投机就约出来见面，关系要从线上走到线下。"),
    ("F48", "关系",
     "老朋友创业找你投资，项目看起来不错，但你俩对钱的事向来敏感。",
     13, [5], "公开合作：钱的事先写清楚，亲兄弟也要明算账。"),
    ("F49", "关系",
     "伴侣总加班到半夜，回家倒头就睡，两人几乎没话说了。",
     8, [4], "互相依靠：先约个固定的时间见面，关系要靠相处养。"),
    ("F50", "关系",
     "邻居装修吵得人睡不着觉，上门说了两次没用，是忍还是投诉？",
     6, [5], "争执：把物业拉进来做中间人，别自己和邻居硬碰。"),
    ("F51", "关系",
     "大学同学聚会，有人炫耀、有人攀比，去了难受、不去又怕被说不合群。",
     15, [2], "功成不居：去不去都行，去了低调点、别跟着攀比。"),
    ("F52", "关系",
     "父母生病住院，兄弟姐妹为谁出钱、谁陪护吵起来了。",
     45, [3], "人聚到一起：先把分工和出钱的标准定下来，再谈感情。"),
    ("F53", "关系",
     "和伴侣冷战三天了，谁也不先开口，其实两人都想和好。",
     58, [2], "坦诚沟通：先开口的那个人不是输了，是把关系救回来了。"),
    ("F54", "关系",
     "职场上被同事抢了功劳，老板还表扬了他，心里憋屈。",
     36, [5], "环境黑暗：先把功劳的证据留好，等合适的时机再说。"),
    ("F55", "关系",
     "孩子想学钢琴，学费不便宜，不知道能不能坚持下来。",
     53, [1], "像大雁一样一步步来：先报个短期的试试，别一下交一年的钱。"),
    ("F56", "关系",
     "老家的房子要不要卖，卖了怕以后回不去，不卖又一直空着。",
     52, [2], "守住边界：先想清楚房子对你到底意味着什么，再决定卖不卖。"),
    ("F57", "关系",
     "朋友圈里有人总爱晒、总爱比，看得人心烦，屏蔽又怕对方发现。",
     20, [2], "先看清楚：屏蔽是小事，别让别人的晒法左右自己的心情。"),
    ("F58", "关系",
     "伴侣想养一只狗，你担心没时间照顾，两人意见不一致。",
     27, [2], "养什么都要算清楚时间账：先商量好分工，再决定养不养。"),
    ("F59", "关系",
     "多年没联系的高中同学突然找你借钱，说周转一下。",
     44, [3], "意外相遇：借可以借，但数额要在自己能承受的范围。"),
    ("F60", "关系",
     "父母想来城里长住，家里住不下，租个房子又怕老人觉得被嫌弃。",
     46, [3], "像树一样慢慢长：先短住试试，住得惯再谈长住。"),
    # ---- 创意 (F61–F80) ----
    ("F61", "创意",
     "写小说写到一半卡住了，主角的动机怎么都圆不上，是改大纲还是硬写下去？",
     18, [1, 5], "内部出了问题：大纲的毛病要从根上改，改到底局面才能重新向上。"),
    ("F62", "创意",
     "拍短视频，内容用心做但没流量，是坚持风格还是追热点？",
     57, [3], "像风一样渗透：热点可以蹭，但风格不能丢，小步推进。"),
    ("F63", "创意",
     "设计一个新 logo，客户说“要大气”，又说不出到底想要什么。",
     20, [4], "先看清楚再做决定：多给几个方向让客户选，别猜。"),
    ("F64", "创意",
     "乐队排练总吵架，吉他手想玩重的，主唱想走流行。",
     38, [3], "看法不同：先找到一首两边都能接受的歌，再谈风格。"),
    ("F65", "创意",
     "做一个播客，第一期录了三遍都不满意，是继续磨还是先发出来？",
     50, [2], "新局面立住了：先发出来，听众的反馈比自己磨更有用。"),
    ("F66", "创意",
     "画了一幅画，自己很满意但没人看得懂，是改还是坚持？",
     30, [3], "像火一样明亮：作品要让人看得见光，先让懂的人看懂。"),
    ("F67", "创意",
     "想做一个没人做过的产品，但不知道有没有市场，是先做出来还是先调研？",
     1, [1, 2], "带头做事：先动起来做个最小的版本，机会和同伴会在路上出现。"),
    ("F68", "创意",
     "写歌写不出副歌，旋律在脑子里转了三天，就是落不下来。",
     29, [4], "一坑接一坑：先放一放，去走走路，旋律会自己落下来。"),
    ("F69", "创意",
     "拍纪录片，主角突然反悔不让拍了，前期投入全白费。",
     47, [3], "被困住了：找个中间人再谈一次，实在不行就换主角。"),
    ("F70", "创意",
     "做游戏，玩法很新颖但测试玩家都说看不懂规则。",
     4, [5], "不懂就问：让玩家告诉你哪里不懂，规则要从玩家的困惑里改。"),
    ("F71", "创意",
     "写剧本，投资人要求加一个流量明星，角色根本不适合。",
     22, [5], "外表光鲜：明星可以加，但角色和故事不能为明星让路。"),
    ("F72", "创意",
     "摄影展的照片选好了，场地突然说档期给了别人。",
     24, [2], "低谷反弹：场地没了就换个地方，作品在、展览就能重新开始。"),
    ("F73", "创意",
     "做手工品牌，订单多了做不过来，是涨价还是找人帮忙？",
     9, [2], "小有收获：先涨一点价把量压下来，别急着扩招。"),
    ("F74", "创意",
     "写公众号，追热点涨粉快，但写得心累，不追又怕掉粉。",
     60, [3], "凡事有度：定个节奏，热点一周最多追一次，剩下的写自己想写的。"),
    ("F75", "创意",
     "编舞编到一半，音乐换了版权方不让用了。",
     49, [2], "旧的要改：音乐换了就换个思路编，变革要顺着音乐走。"),
    ("F76", "创意",
     "做播客想请个大咖，对方一直不回消息，是继续约还是放弃？",
     5, [2], "时机未到：先做几期好的内容出来，再去约更有底气。"),
    ("F77", "创意",
     "小说里两个角色的感情线写崩了，读者都在骂。",
     31, [5], "两心相感：感情线要从人物的真实动机里长出来，别硬拗。"),
    ("F78", "创意",
     "做独立游戏，美术风格定了三版都不满意，时间却不多了。",
     21, [2], "中间有梗阻：定个 deadline，选最接近的那版，别无限改下去。"),
    ("F79", "创意",
     "拍 vlog，剪出来的片子自己看着都困，是重剪还是换题材？",
     16, [3], "心情松快：先想清楚观众为什么会困，再动手重剪。"),
    ("F80", "创意",
     "写诗投稿被退了三次，编辑说“感觉不对”，又不说哪里不对。",
     52, [3], "该停就停：先放一阵子，回头再读自己的诗，问题会自己冒出来。"),
    # ---- 日常 (F81–F100) ----
    ("F81", "日常",
     "连续熬夜一周，身体开始抗议，但手头的事一件都没做完。",
     27, [3, 5], "靠什么养自己：先睡一觉把根本养好，事情可以晚一天做。"),
    ("F82", "日常",
     "减肥减了两个月，体重一点没动，是方法错了还是坚持得不够？",
     32, [3], "长久坚持：先检查方法对不对，节奏对了才谈坚持。"),
    ("F83", "日常",
     "手机用了三年，电池不行了，换电池还是换新机？",
     41, [2], "先舍后得：换电池是小钱，换新机是大钱，先算清楚再决定。"),
    ("F84", "日常",
     "周末想出去玩，但家里一堆事没做完，去还是不去？",
     52, [4], "该停就停：家里的事做不完就先停半天，出去透透气回来再做。"),
    ("F85", "日常",
     "存了点钱，是先还房贷还是留着应急？",
     60, [2], "凡事有度：留够应急的钱，剩下的再还房贷。"),
    ("F86", "日常",
     "孩子发烧 38 度，是去医院还是在家观察？",
     29, [3], "一坑接一坑：先看精神状态，精神好就在家观察，精神差马上送医院。"),
    ("F87", "日常",
     "老家房子漏水，修一次要不少钱，不修又越漏越厉害。",
     18, [2, 4], "内部出了问题：漏水不修只会越拖越贵，先从基层动手，再把整顿做到底。"),
    ("F88", "日常",
     "每天通勤两小时，累得没时间陪家人，是换工作还是搬家？",
     56, [5], "在外漂泊：先算清楚换工作和搬家各自的代价，再选。"),
    ("F89", "日常",
     "想学一门新手艺，报了课又总没时间去，钱白花了。",
     53, [3], "像大雁一样一步步来：先每周固定一个小时，能坚持再加量。"),
    ("F90", "日常",
     "家里的旧东西堆满了阳台，扔舍不得、留着占地方。",
     41, [5], "减掉多余的才能轻装上路：先扔掉一年没用过的。"),
    ("F91", "日常",
     "体检报告出来几项指标偏高，医生说注意饮食，但应酬推不掉。",
     27, [5], "言语和饮食都要节制：应酬能推就推，推不掉就少吃多动。"),
    ("F92", "日常",
     "下雨天开车上班，路上堵得一动不动，是等还是绕路？",
     39, [2], "走不动了：看看导航有没有小路，别在一条路上死等。"),
    ("F93", "日常",
     "想给父母买个礼物，买贵的怕他们心疼，买便宜的又怕不够好。",
     15, [3], "功成不居：礼物贵贱不重要，合心意、让父母安心才重要。"),
    ("F94", "日常",
     "每天刷手机到半夜，第二天起不来，想戒又戒不掉。",
     52, [1], "该停就停：先定个睡觉前一小时不碰手机的规矩。"),
    ("F95", "日常",
     "菜市场买菜，总觉得被缺斤少两，是换个摊还是当面说？",
     61, [2], "心中有诚信：先当面客气地提一次，再有下次就换摊。"),
    ("F96", "日常",
     "冬天到了，暖气还没来，家里冷得睡不着，是买电暖器还是忍几天？",
     5, [5], "时机未到：先买个小的顶几天，暖气来了就不用了。"),
    ("F97", "日常",
     "朋友推荐了一只股票，说肯定涨，买还是不买？",
     25, [1, 3], "不妄动：按本分做事，别听风就是雨，先想清楚自己能不能承受。"),
    ("F98", "日常",
     "家里养的花总养不活，换了几盆还是死。",
     48, [2], "养好根本：先弄明白是水多了还是光少了，再谈养花。"),
    ("F99", "日常",
     "想跑步锻炼，跑两天就放弃了，每次都是这样。",
     32, [5], "长久坚持：别定每天五公里，先定每天十分钟。"),
    ("F100", "日常",
     "年底了，想给自己定个新年目标，定高了怕完不成，定低了又觉得没劲。",
     26, [2, 6], "积蓄力量：目标定在踮脚够得着的地方，蓄满了再往上加。"),
]


def _evolution(start_kw: int, lines: list[int]) -> dict:
    """Proposed evolution with the end hexagram COMPUTED, never hand-typed."""
    by_kw = {h.king_wen: h for h in load_hexagrams()}
    by_bits = {h.bits: h for h in load_hexagrams()}
    start = by_kw[start_kw]
    bits = start.bits
    for line in lines:
        bits = flip_bit(bits, line)
    end = by_bits[bits]
    return {
        "start_king_wen": start.king_wen,
        "start_name": start.name_zh,
        "start_bits": start.bits,
        "moving_lines": list(lines),
        "end_king_wen": end.king_wen,
        "end_name": end.name_zh,
        "end_bits": end.bits,
    }


def build_pass_a() -> list[dict]:
    """Pass A labels from the SCENARIOS table."""
    out = []
    for sid, category, text, start_kw, lines, rationale in SCENARIOS:
        evo = _evolution(start_kw, lines)
        evo["rationale"] = rationale
        out.append({"id": sid, "category": category, "scenario": text, "pass_a": evo})
    return out


def run_pass_b(entries: list[dict], encoder: GroundingEncoder) -> list[dict]:
    """Pass B: embedding-resonance top-1 + v1 reasoner path, per scenario.

    A different method from pass A (retrieval, not reasoning from
    meanings). Pass B never overrides pass A; it only confirms or
    flags. The manual check of the disagreements is done by the author
    after this runs (see docs/exam-100-notes.md).
    """
    for entry in entries:
        r = reason(entry["scenario"], encoder)
        evo = _evolution(r.start_king_wen, [s.line for s in r.steps])
        entry["pass_b"] = {
            **evo,
            "method": "embedding-resonance top-1 + v1 reasoner path (rich documents)",
            "top1_score": r.match_score,
        }
    return entries


def finalize(entries: list[dict]) -> list[dict]:
    """Final label per scenario: agreement, or pass A on a tie (C pending).

    With pass C pending there is no majority on a disagreement, so the
    tie-break takes pass A and flags the scenario for human review.
    """
    for entry in entries:
        a, b = entry["pass_a"], entry["pass_b"]
        start_agree = a["start_king_wen"] == b["start_king_wen"]
        lines_agree = a["moving_lines"] == b["moving_lines"]
        full_agree = start_agree and lines_agree
        entry["agreement"] = {
            "start_agree": start_agree,
            "lines_agree": lines_agree,
            "full_agree": full_agree,
        }
        entry["flagged"] = not full_agree
        entry["final_source"] = (
            "A+B agree" if full_agree else "A (tie-break; pass C pending human review)"
        )
        # "proposed" is the final label, in the exam_v1.json shape, so
        # the existing harness (taiji_field.exam.run_exam) runs unchanged.
        proposed = {k: a[k] for k in (
            "start_king_wen", "start_name", "start_bits", "moving_lines",
            "end_king_wen", "end_name", "end_bits", "rationale",
        )}
        entry["proposed"] = proposed
        entry["label_status"] = (
            "draft — pass A (author) + pass B (embedding resonance); "
            "pass C pending human review; NOT ground truth"
        )
    return entries


def build() -> dict:
    assert len(SCENARIOS) == 100, f"expected 100 scenarios, got {len(SCENARIOS)}"
    from collections import Counter
    counts = Counter(cat for _, cat, *_ in SCENARIOS)
    assert dict(counts) == {c: 20 for c in CATEGORIES}, dict(counts)
    assert len({s[0] for s in SCENARIOS}) == 100, "scenario ids must be unique"
    encoder = GroundingEncoder()
    print(f"[pass B encoder backend: {encoder.backend}]")
    entries = finalize(run_pass_b(build_pass_a(), encoder))
    n_start = sum(e["agreement"]["start_agree"] for e in entries)
    n_full = sum(e["agreement"]["full_agree"] for e in entries)
    n_lines = sum(e["agreement"]["lines_agree"] for e in entries)
    return {
        "version": "100-v1",
        "note": (
            "DRAFT labels. Consensus is 2-agent + (1-LLM or pending-human), "
            "NOT 3-human — a planned deviation from roadmap section 6, to be "
            "upgraded when humans review. Pass A: author reasoning from "
            "hexagram meanings. Pass B: embedding-resonance retrieval + "
            "manual check of disagreements (a different method from A, not a "
            "copy). Pass C: pending human review — no second LLM is available "
            "through the connectors, and the baseline model (DeepSeek "
            "deepseek-chat) must not also be the annotator. Where A and B "
            "disagree, the tie-break takes pass A and flags the scenario. "
            "End hexagrams are computed from start + moving lines, never "
            "hand-typed."
        ),
        "annotation": {
            "pass_A": "author reasoning from hexagram meanings (start + moving lines + rationale)",
            "pass_B": "embedding-resonance top-1 + v1 reasoner path, disagreements manually checked by the author",
            "pass_C": "pending-human-review — no second general LLM available via connectors (meta-model-api is FixLens photo analysis only); baseline model excluded as annotator to avoid bias",
            "tie_break": "on A/B disagreement, take pass A and flag the scenario",
        },
        "agreement_summary": {
            "n": 100,
            "start_agree": n_start,
            "lines_agree": n_lines,
            "full_agree": n_full,
            "flagged": 100 - n_full,
        },
        "scenarios": entries,
    }


def main() -> None:
    data = build()
    json_path = ROOT / "taiji_field" / "data" / "exam_100.json"
    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    s = data["agreement_summary"]
    print(f"wrote {json_path} (100 scenarios)")
    print(f"agreement: start {s['start_agree']}/100, lines {s['lines_agree']}/100, "
          f"full {s['full_agree']}/100, flagged {s['flagged']}/100")


if __name__ == "__main__":
    main()
