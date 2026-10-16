# -*- coding: utf-8 -*-
"""生成《广告素材合规审核系统》汇报 PPT。

设计：浅色学术风（白底 + 深蓝主色 + 灰阶），16:9，每页写讲者备注。
数据：全部取自本项目实测与 04_Bett/实验报告.html，本文件里不出现任何编造的指标。

用法：
    <python> 07-report/build_ppt.py
"""
import pathlib
import sys

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

sys.stdout.reconfigure(encoding="utf-8")

OUT = pathlib.Path(__file__).parent / "广告素材合规审核系统_汇报.pptx"

# ---------------------------------------------------------------- 主题
INK = RGBColor(0x1F, 0x23, 0x29)        # 主文字
MUT = RGBColor(0x6B, 0x72, 0x80)        # 次要文字
DIM = RGBColor(0x9C, 0xA3, 0xAF)        # 更弱
ACCENT = RGBColor(0x18, 0x5F, 0xA5)     # 深蓝主色
ACCENT_L = RGBColor(0xEA, 0xF1, 0xF8)   # 主色浅底
TEAL = RGBColor(0x0F, 0x6E, 0x56)       # 青绿（正向指标）
RED = RGBColor(0xA3, 0x2D, 0x2D)        # 危险
AMBER = RGBColor(0x85, 0x4F, 0x0B)      # 警示
LINE = RGBColor(0xE5, 0xE7, 0xEB)
CARD = RGBColor(0xF6, 0xF7, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "微软雅黑"

SW, SH = Inches(13.333), Inches(7.5)
M = Inches(0.62)                        # 页边距


def new_deck():
    prs = Presentation()
    prs.slide_width, prs.slide_height = SW, SH
    return prs


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def textbox(slide, x, y, w, h, text, size=14, color=INK, bold=False,
            align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.15):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = FONT
    return box


def rect(slide, x, y, w, h, fill=CARD, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(0.75)
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        try:
            s.adjustments[0] = radius
        except Exception:
            pass
    return s


def header(slide, kicker, title, sub=None):
    """统一的页眉：小节标签 + 标题 + 主色细线。"""
    textbox(slide, M, Inches(0.34), Inches(11), Inches(0.3), kicker, size=11, color=ACCENT, bold=True)
    textbox(slide, M, Inches(0.62), Inches(12.1), Inches(0.6), title, size=25, bold=True)
    ln = rect(slide, M, Inches(1.26), Inches(1.5), Pt(2.6), fill=ACCENT, shape=MSO_SHAPE.RECTANGLE)
    return ln


def footer(slide, text):
    textbox(slide, M, Inches(6.98), Inches(12.1), Inches(0.32), text, size=10, color=DIM)


def style_table(tbl, col_widths=None, header_fill=ACCENT, size=11, header_size=11):
    tbl.first_row = False
    tbl.horz_banding = False
    if col_widths:
        for i, w in enumerate(col_widths):
            tbl.columns[i].width = w
    for r, row in enumerate(tbl.rows):
        for c, cell in enumerate(row.cells):
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = header_fill
            else:
                cell.fill.fore_color.rgb = WHITE if r % 2 else CARD
            for p in cell.text_frame.paragraphs:
                p.line_spacing = 1.05
                for run in p.runs:
                    run.font.name = FONT
                    run.font.size = Pt(header_size if r == 0 else size)
                    run.font.bold = (r == 0)
                    run.font.color.rgb = WHITE if r == 0 else INK
    return tbl


def table_slide(prs, kicker, title, headers, rows, col_widths, note=None,
                size=11, row_h=Inches(0.34), head_h=Inches(0.36), y=Inches(1.62),
                note_text=None):
    slide = blank(prs)
    header(slide, kicker, title)
    n_rows, n_cols = len(rows) + 1, len(headers)
    total_w = sum(col_widths)
    gf = slide.shapes.add_table(n_rows, n_cols, M, y, total_w, head_h + row_h * len(rows))
    tbl = gf.table
    for c, htxt in enumerate(headers):
        tbl.cell(0, c).text = str(htxt)
    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            tbl.cell(r, c).text = str(val)
    tbl.rows[0].height = head_h
    for r in range(1, n_rows):
        tbl.rows[r].height = row_h
    style_table(tbl, col_widths, size=size)
    if note:
        note_box(slide, y + head_h + row_h * len(rows) + Inches(0.16), note)
    return slide


def note_box(slide, y, text, color=ACCENT, fill=ACCENT_L, h=Inches(0.52)):
    box = rect(slide, M, y, Inches(12.1), h, fill=fill)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.14)
    tf.margin_right = Inches(0.12)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.line_spacing = 1.1
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = text
    run.font.size = Pt(11)
    run.font.color.rgb = color
    run.font.name = FONT
    return box


# ================================================================ 1. 封面
prs = new_deck()
s = blank(prs)
rect(s, Inches(0), Inches(0), Inches(0.34), SH, fill=ACCENT, shape=MSO_SHAPE.RECTANGLE)
textbox(s, Inches(1.15), Inches(1.95), Inches(11.2), Inches(1.0),
        "广告素材合规审核系统", size=44, bold=True)
textbox(s, Inches(1.15), Inches(3.02), Inches(11.2), Inches(0.5),
        "中文短文本六分类 · 从业务需求到轻量化部署", size=19, color=MUT)
rect(s, Inches(1.18), Inches(3.72), Inches(2.0), Pt(3), fill=ACCENT, shape=MSO_SHAPE.RECTANGLE)
textbox(s, Inches(1.15), Inches(4.05), Inches(11.2), Inches(1.4),
        "四条技术路线对比（随机森林 / fastText / BERT 微调 / 大模型）\n"
        "置信度拒判 · 动态量化 · 知识蒸馏 · 非结构化剪枝\n"
        "四路线 API 服务 + 单文件前端审核工作台", size=14, color=MUT, spacing=1.5)
textbox(s, Inches(1.15), Inches(6.15), Inches(11.2), Inches(0.4),
        "汇报人：____________        日期：____________", size=13, color=DIM)
notes(s, "开场（约 30 秒）：\n"
         "一句话说清：把「一段广告文案 → 六类合规判定」做成可上线的模型服务，"
         "并且**模型没把握时会主动转人工**，而不是硬判。\n"
         "四条路线都实际跑通了：随机森林、fastText、BERT 微调、大模型，"
         "最终选 BERT 加权 + 蒸馏/量化做部署。\n"
         "提醒自己：全程用真实实验数据，测试集 956 条，不吹指标。")

# ================================================================ 2. 汇报路线
s = blank(prs)
header(s, "汇报路线", "六个板块")
items = [
    ("一、业务需求分析", "场景（What）· 需求（How）· 量化目标"),
    ("二、项目整体架构", "数据 → 训练 → 优化 → 服务 四层"),
    ("三、数据说明", "来源 · 规模 · 标签分布 · 清洗策略"),
    ("四、模型方案对比", "Baseline · 改进方案 · 消融实验"),
    ("五、最终模型方案", "模型架构 · 评测指标"),
    ("六、模型部署", "压缩方法 · 接口设计 · 前端设计"),
]
x0, y0 = M, Inches(1.72)
cw, ch, gap = Inches(3.92), Inches(1.42), Inches(0.18)
for i, (t, sub) in enumerate(items):
    r, c = divmod(i, 3)
    x = x0 + c * (cw + gap)
    y = y0 + r * (ch + Inches(0.22))
    rect(s, x, y, cw, ch, fill=CARD)
    rect(s, x, y, Pt(3.2), ch, fill=ACCENT, shape=MSO_SHAPE.RECTANGLE)
    textbox(s, x + Inches(0.22), y + Inches(0.2), cw - Inches(0.4), Inches(0.35), t, size=15, bold=True)
    textbox(s, x + Inches(0.22), y + Inches(0.62), cw - Inches(0.4), Inches(0.6), sub, size=11.5, color=MUT, spacing=1.3)
footer(s, "每一步都给真实数字；口径有歧义的地方我会明确标注来源")
notes(s, "这一页只花 20 秒，让大家知道后面有六块、按顺序讲。\n"
         "可以补一句：'最后一页我会把容易混淆的口径单独说清楚，避免大家记错数字。'")

# ================================================================ 3. 业务场景
s = blank(prs)
header(s, "一、业务需求分析", "业务场景是什么（What）")
rect(s, M, Inches(1.62), Inches(12.1), Inches(1.15), fill=CARD)
textbox(s, M + Inches(0.24), Inches(1.78), Inches(11.6), Inches(0.85),
        "广告素材上线投放前必须做合规审核。传统做法是人工全量审核 —— 成本高、口径不一致、产能有瓶颈。\n"
        "现在要做的是「机器初筛 + 置信度分流」：符合标准的自动放行或拦截，拿不准的转人工。",
        size=13.5, spacing=1.4)
blocks = [
    ("每天的量", "平台持续产生大量投放文案与素材文本，需要逐条过审", ACCENT),
    ("在线实时", "单条审核要在秒级返回；同时支持批量过一遍历史素材", TEAL),
    ("不只是分类", "结果还要作为投放风控的「内容标签」，供后续策略使用", AMBER),
]
bw = Inches(3.92)
for i, (t, d, col) in enumerate(blocks):
    x = M + i * (bw + Inches(0.18))
    rect(s, x, Inches(3.0), bw, Inches(1.75), fill=WHITE, line=LINE)
    rect(s, x, Inches(3.0), bw, Pt(3), fill=col, shape=MSO_SHAPE.RECTANGLE)
    textbox(s, x + Inches(0.22), Inches(3.2), bw - Inches(0.4), Inches(0.34), t, size=15, bold=True, color=col)
    textbox(s, x + Inches(0.22), Inches(3.62), bw - Inches(0.4), Inches(1.0), d, size=12, color=MUT, spacing=1.35)
note_box(s, Inches(5.0), "业务痛点一句话：全量人工审不动，但漏放违规的代价 >> 多审几条的代价 —— 这条决定了后面所有技术取舍。",
         h=Inches(0.6))
notes(s, "要点：把'场景'讲成业务方听得懂的话。\n"
         "必说的两句：① 人工全量审不动（成本/口径/产能）；② **漏放违规的代价远大于误判**——"
         "所以后面我会为了提升少数类召回，主动牺牲一点整体精确率。\n"
         "可能被问：'真实违规占比多少？' 答：实验集里违规占近 90%，真实投放远低于此，"
         "这是本项目的已知局限（最后会讲），阈值需要按真实先验重新标定。")

# ================================================================ 4. 业务需求
s = blank(prs)
header(s, "一、业务需求分析", "业务需求是什么（How）")
rows = [
    ("建模方式", "单标签文本分类：一段广告文案 → 六个类别之一"),
    ("类别体系", "不违规 / 黑产广告 / 欺诈 / 博彩 / 谩骂引战 / 低俗色情（顺序即标签索引，固定不变）"),
    ("关键业务约束", "「没把握」必须是合法输出：置信度 < 阈值 → 转人工复核，而不是强行猜一个类别"),
    ("工程需求", "接口化部署（HTTP API）、支持批量处理、可切换多条技术路线做对比"),
    ("可解释要求", "除类别外，还要给出置信度与六类概率分布，便于人工复核时参考"),
]
y = Inches(1.66)
for i, (k, v) in enumerate(rows):
    rect(s, M, y, Inches(12.1), Inches(0.74), fill=CARD if i % 2 == 0 else WHITE,
         line=None if i % 2 == 0 else LINE)
    textbox(s, M + Inches(0.22), y + Inches(0.19), Inches(2.1), Inches(0.4), k, size=13, bold=True, color=ACCENT)
    textbox(s, M + Inches(2.45), y + Inches(0.17), Inches(9.4), Inches(0.5), v, size=12.5, spacing=1.25)
    y += Inches(0.8)
note_box(s, Inches(5.78), "拒判机制是本项目与「普通文本分类作业」最大的区别：argmax 永远会挑一个类别出来，"
                          "但真实审核里「我不知道」本身就是一个有效答案。", h=Inches(0.66))
notes(s, "重点讲第三条（关键业务约束）——这是整个方案的技术锚点。\n"
         "话术：'普通分类任务的输出是六选一，但审核业务需要第七种输出：转人工。'"
         "后面你会看到，这条约束直接催生了阈值实验、也让 BERT 相对 fastText 的优势成立。\n"
         "第四条工程需求可以顺势带出'我做了四条路线 + 一个统一前端'。")

# ================================================================ 5. 量化目标
table_slide(
    prs, "一、业务需求分析", "业务目标（量化指标）：目标 vs 实测",
    ["指标", "目标", "实测", "判定"],
    [
        ("测试集 macro-F1", "≥ 0.90", "0.9343", "达标"),
        ("测试集 准确率", "≥ 0.92", "0.9310", "达标"),
        ("少数类（欺诈）召回", "≥ 0.90", "0.94", "达标"),
        ("自动判定占比", "≥ 90%", "96.0%（阈值 0.60）", "达标"),
        ("主模型体积", "≤ 500MB", "390.2MB", "达标"),
        ("边缘侧单条延迟（模型前向）", "≤ 2ms", "1.68ms（蒸馏 Student）", "达标"),
    ],
    [Inches(4.3), Inches(1.9), Inches(3.6), Inches(2.3)],
    note=None, size=12, row_h=Inches(0.44),
)
s = prs.slides[-1]
note_box(s, Inches(4.85),
         "口径提醒：准确率与 macro-F1 采用 04_Bett/实验报告 口径（0.9310 / 0.9343）。"
         "README 里写的「准确率 0.9331」实为 dev 验证集的 val_f1，不是准确率，口径问题最后一页会专门说。",
         color=AMBER, fill=RGBColor(0xFF, 0xF9, 0xEC), h=Inches(0.72))
notes(s, "逐行念完这张表就是最有力的开场——所有目标都达标。\n"
         "两个必须主动交代的点：\n"
         "1) 准确率 0.9310 与 macro-F1 0.9343 的差别：macro-F1 略高说明各类更均衡，"
         "这是类别加权带来的效果。\n"
         "2) 底部的口径提醒一定要说——README 的 0.9331 是 **val_f1**，"
         "而且来自 dev 验证集（947 条），不是测试集。老师如果看过仓库 README 会问这个。\n"
         "如果被问'边缘侧是什么设备'：答：CPU 推理场景（无 GPU 的服务器/边缘盒子），"
         "指标用蒸馏后 Student 的模型前向耗时 1.68ms 与 45.6M 参数量衡量。")

# ================================================================ 6. 整体架构
s = blank(prs)
header(s, "二、项目整体架构", "四层链路：数据 → 训练 → 优化 → 服务")
layers = [
    ("数据层", "原始数据 → 清洗去重 → 分层切分 → 统一文本契约（文本\\t标签）", ACCENT),
    ("训练层", "四条技术路线并行：随机森林(TF-IDF) · fastText(字/词级) · BERT 微调(主路线) · 大模型提示词", TEAL),
    ("优化层", "类别加权 · 置信度拒判 · 动态量化(体积) · 知识蒸馏(层数/延迟) · 非结构化剪枝(gzip 体积)", AMBER),
    ("服务层", "四路线 HTTP API（各自的端口） + 单文件前端审核工作台 + 转人工分流", RED),
]
y = Inches(1.7)
for name, desc, col in layers:
    rect(s, M, y, Inches(12.1), Inches(1.0), fill=CARD)
    rect(s, M, y, Inches(1.55), Inches(1.0), fill=col, shape=MSO_SHAPE.RECTANGLE)
    textbox(s, M, y + Inches(0.3), Inches(1.55), Inches(0.4), name, size=15, bold=True,
            color=WHITE, align=PP_ALIGN.CENTER)
    textbox(s, M + Inches(1.78), y + Inches(0.2), Inches(10.1), Inches(0.65), desc, size=12.5,
            color=INK, spacing=1.3, anchor=MSO_ANCHOR.MIDDLE)
    y += Inches(1.12)
note_box(s, Inches(6.24), "工程上的一处关键改造：四条路线原来是四个各自独立的 Streamlit 演示页，"
                          "现在统一成「同契约 API + 同一个前端工作台」，可现场横向对比。", h=Inches(0.6))
notes(s, "这一页给全局，别陷细节。每层一句话：\n"
         "数据层：只强调'统一文本契约'——后面所有路线都吃同一份 train/dev/test。\n"
         "训练层：强调四条路线**都真的跑起来了**，不是只写了个方案。\n"
         "优化层：点出量化/蒸馏/剪枝各解决不同问题（体积 vs 延迟 vs 稀疏度）。\n"
         "服务层：统一契约 + 一个前端，这是后面部署部分要展开的。")

# ================================================================ 7. 数据来源与规模
s = blank(prs)
header(s, "三、数据说明", "来源与规模")
kpis = [
    ("9503 条", "去重后总样本", ACCENT),
    ("7600 / 947 / 956", "train / dev / test（按类分层 8:1:1）", TEAL),
    ("17 字", "文本长度中位数（均值 37.5 字）", AMBER),
    ("6 类", "标签体系，顺序固定不可变", RED),
]
bw = Inches(2.92)
for i, (v, k, col) in enumerate(kpis):
    x = M + i * (bw + Inches(0.16))
    rect(s, x, Inches(1.66), bw, Inches(1.3), fill=CARD)
    textbox(s, x + Inches(0.16), Inches(1.82), bw - Inches(0.3), Inches(0.5), v, size=21, bold=True, color=col)
    textbox(s, x + Inches(0.16), Inches(2.38), bw - Inches(0.3), Inches(0.5), k, size=11.5, color=MUT, spacing=1.25)
rows = [
    ("来源", "zjunlp/ChineseHarm-bench（HuggingFace 公开学术数据集，中文有害内容检测基准）"),
    ("原始字段", "文本 + 标签；转换为项目统一契约：一行一条「文本\\t标签索引」，LF 行尾"),
    ("切分方式", "按类别分层抽样（stratify），保证 train/dev/test 六类比例一致，随机种子固定 42"),
    ("max_len 依据", "中位数 17 字、均值 37.5 字、p95 129 字 → 取 max_len = 64 覆盖绝大多数样本"),
]
y = Inches(3.18)
for i, (k, v) in enumerate(rows):
    rect(s, M, y, Inches(12.1), Inches(0.72), fill=WHITE if i % 2 else CARD)
    textbox(s, M + Inches(0.22), y + Inches(0.18), Inches(1.7), Inches(0.4), k, size=12.5, bold=True, color=ACCENT)
    textbox(s, M + Inches(2.05), y + Inches(0.16), Inches(9.8), Inches(0.5), v, size=12, spacing=1.25)
    y += Inches(0.78)
notes(s, "数据这块讲三件事：哪来的、多大、为什么这么切。\n"
         "主动说清：**max_len=64 不是我拍的**，是看了长度分布（中位数 17、p95 129）定的。\n"
         "如果被问'为什么用公开数据集而不是公司数据'：答：这是教学项目，"
         "用公开基准保证可复现；真实业务需要按自己的标签体系重新标注，这点在局限里提。")

# ================================================================ 8. 标签分布（图）
s = blank(prs)
header(s, "三、数据说明", "标签分布：六类占比与不平衡度")
cd = CategoryChartData()
cd.categories = ["不违规", "欺诈", "博彩", "黑产广告", "谩骂引战", "低俗色情"]
cd.add_series("样本数", (988, 1000, 1748, 1867, 1904, 1996))
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, M, Inches(1.62),
                        Inches(7.5), Inches(4.5), cd)
ch = gf.chart
ch.has_legend = False
ch.font.size = Pt(11)
ch.font.name = FONT
plot = ch.plots[0]
plot.has_data_labels = True
plot.data_labels.font.size = Pt(11)
plot.data_labels.font.color.rgb = INK
plot.gap_width = 55
plot.series[0].format.fill.solid()
plot.series[0].format.fill.fore_color.rgb = ACCENT

rect(s, Inches(8.35), Inches(1.62), Inches(4.37), Inches(2.1), fill=CARD)
textbox(s, Inches(8.55), Inches(1.78), Inches(4.0), Inches(1.8),
        "不平衡比  2.02 : 1\n最大类 低俗色情 21.0%（1996 条）\n最小类 不违规 10.4%（988 条）\n少数类 欺诈 仅 10.5%（1000 条）",
        size=12.5, spacing=1.5, color=INK)
rect(s, Inches(8.35), Inches(3.86), Inches(4.37), Inches(2.26), fill=RGBColor(0xFF, 0xF9, 0xEC))
textbox(s, Inches(8.55), Inches(4.02), Inches(4.0), Inches(1.95),
        "为什么分布值得单独讲：\n"
        "① 欺诈只占 10.5%，是后面召回率翻车的根源；\n"
        "② 「不违规」只有 10.4%，而真实投放里违规是极少数 —— 先验分布不同，是本项目最大的局限。",
        size=11.5, spacing=1.4, color=AMBER)
footer(s, "数据来源：01-data/train.txt + dev.txt + test.txt 全量实测统计（9503 条）")
notes(s, "图表是真实统计（我逐行读过三个文件）。\n"
         "这一页要埋两个伏笔：\n"
         "1) 欺诈类最少 → 下一页/后面的类别加权就是为了救它（召回 0.77 → 0.94）。\n"
         "2) 不违规只占 10.4% —— 说一句'这个分布和真实投放很不一样'，"
         "为最后的局限铺垫，也显示你知道自己的数据边界。")

# ================================================================ 9. 清洗策略
s = blank(prs)
header(s, "三、数据说明", "数据清洗策略：六步流水线")
steps = [
    ("1 空白符归一", "正则把 \\t \\r \\n 统一替换为空格，保证「一行一条」的契约不被破坏"),
    ("2 过短样本丢弃", "长度 < 2 字的样本视为噪声（多为标注残留）"),
    ("3 完全重复去重", "同一 (文本, 标签) 只保留一条"),
    ("4 标签冲突丢弃", "同一文本被标成不同类别 → 整条丢弃，这类是标注噪声，留着会污染训练与评估"),
    ("5 按类分层切分", "8:1:1 分层抽样，固定随机种子 42，保证三份数据类别比例一致"),
    ("6 固定落盘契约", "文本\\t标签索引，显式 LF 行尾（避免 Windows 默认 CRLF 让标签带上 \\r）"),
]
y = Inches(1.66)
for i, (t, d) in enumerate(steps):
    rect(s, M, y, Inches(12.1), Inches(0.72), fill=CARD if i % 2 == 0 else WHITE)
    textbox(s, M + Inches(0.22), y + Inches(0.18), Inches(2.3), Inches(0.4), t, size=12.5, bold=True, color=ACCENT)
    textbox(s, M + Inches(2.65), y + Inches(0.16), Inches(9.2), Inches(0.5), d, size=12, spacing=1.25)
    y += Inches(0.78)
note_box(s, Inches(6.36), "第 4 步最容易被忽略也最值钱：标签冲突样本是标注噪声，"
                          "留着会让模型学到互相矛盾的信号，指标虚低还找不到原因。", h=Inches(0.55))
notes(s, "清洗不要逐条念，重点讲第 4、6 步：\n"
         "第 4 步（标签冲突丢弃）：这是'数据质量优先'的体现，"
         "可以补一句'我统计过冲突样本数量，它们确实是噪声'。\n"
         "第 6 步（LF 行尾）：这是个真实的坑——Windows 下默认会写成 CRLF，"
         "下游 split('\\t') 取到的标签会带 \\r，int() 直接报错。讲这个会显得你真的跑过代码。")

# ================================================================ 10. Baseline
table_slide(
    prs, "四、模型方案对比", "Baseline 设计及效果",
    ["方案", "关键配置", "accuracy", "macro-F1", "单条推理(CPU)"],
    [
        ("随机森林 + TF-IDF", "n_estimators=200，jieba 前 20 词", "0.784", "0.780", "<1 ms"),
        ("fastText 字级（手动）", "dim 32 / wordNgrams 2 / epoch 100", "—", "0.908", "<1 ms"),
        ("fastText 词级（手动）", "jieba 分词", "—", "0.872", "<1 ms"),
        ("fastText 字级（自动调参）", "autotune 30s（结果有波动）", "—", "0.894~0.909", "<1 ms"),
        ("fastText 词级（自动调参）", "体积上限 50M", "—", "0.855", "<1 ms"),
    ],
    [Inches(2.9), Inches(4.1), Inches(1.5), Inches(2.1), Inches(1.5)],
    size=11.5, row_h=Inches(0.42),
)
s = prs.slides[-1]
note_box(s, Inches(4.55),
         "两点结论：① 传统方法快 100 倍以上，但 F1 低约 2.6 个点；"
         "② 随机森林用的是它自带的 8:2 切分（非统一测试集），这一行不能与其它行直接横向比较。",
         color=AMBER, fill=RGBColor(0xFF, 0xF9, 0xEC), h=Inches(0.7))
notes(s, "Baseline 的作用是'衬托'，但要客观：\n"
         "fastText 字级 0.908 其实相当能打，速度还快 100 倍——"
         "所以说'如果只追求吞吐，fastText 做粗筛完全可用'，这比一味贬低基线更可信。\n"
         "**必须主动交代随机森林的口径问题**：它用的是自带 8:2 切分，不是统一测试集，"
         "0.784 不能和 0.908 直接比。老师很可能抓这一点，先说掉。")

# ================================================================ 11. 改进方案
s = blank(prs)
header(s, "四、模型方案对比", "改进方案：BERT 微调 + 类别加权")
left = [
    ("模型", "bert-base-chinese（12 层，102.3M 参数）+ 单层分类头"),
    ("超参", "max_len 64 · batch 8 · lr 2e-5 · epochs 3 · weight_decay 1e-2"),
    ("关键改进", "类别加权 CrossEntropy，权重 = √(N / (K·count))，用开根号缓和过度补偿"),
    ("实测权重", "不违规 1.27 · 欺诈 1.26 · 黑产广告 0.92 · 博彩 0.95"),
]
y = Inches(1.68)
for k, v in left:
    rect(s, M, y, Inches(7.05), Inches(1.06), fill=CARD)
    textbox(s, M + Inches(0.2), y + Inches(0.14), Inches(1.4), Inches(0.3), k, size=12, bold=True, color=ACCENT)
    textbox(s, M + Inches(0.2), y + Inches(0.44), Inches(6.6), Inches(0.55), v, size=11.5, color=INK, spacing=1.25)
    y += Inches(1.16)

rect(s, Inches(7.95), Inches(1.68), Inches(4.77), Inches(2.0), fill=ACCENT_L)
textbox(s, Inches(8.15), Inches(1.82), Inches(4.4), Inches(1.8),
        "效果（对比无加权基线）\naccuracy  0.9330 → 0.9310（略降）\nmacro-F1  0.930  → 0.9343（+0.4 点）\n欺诈召回  0.77   → 0.94（+17 点）",
        size=12.5, spacing=1.5, color=ACCENT)
rect(s, Inches(7.95), Inches(3.84), Inches(4.77), Inches(2.28), fill=CARD)
textbox(s, Inches(8.15), Inches(3.98), Inches(4.4), Inches(2.05),
        "代价（主动接受）\n"
        "谩骂引战 F1  0.934 → 0.912\n"
        "低俗色情 F1  0.920 → 0.895\n\n"
        "业务判断：漏放欺诈的代价远高于多审几条低俗色情，这笔交换划算。",
        size=11.5, spacing=1.4, color=INK)
footer(s, "注：整体准确率略降 0.2 个点，是主动用「整体精确率」换「少数类召回」的结果，下一张表逐类看")
notes(s, "这一页是全文的技术核心之一。\n"
         "讲法：先说'普通 CrossEntropy 在 2:1 不平衡下会牺牲少数类'，"
         "再给数字：欺诈召回只有 0.77，而审核业务最不能漏的就是欺诈。\n"
         "然后给解法：类别加权，权重用 √(N/(K·count)) 而不是直接用 N/count，"
         "因为**开根号能缓和过度补偿**（直接反比会把多数类压得太狠）。\n"
         "**务必主动讲代价**：谩骂引战和低俗色情的 F1 都掉了，整体准确率也略降 0.2 个点。"
         "能讲清'为什么这笔交换划算'，比只报好消息加分得多。")

# ================================================================ 12. 消融 1：类别加权
table_slide(
    prs, "四、模型方案对比", "消融实验 ①：类别加权逐类影响",
    ["类别", "F1 · 基线", "F1 · 加权", "召回 · 加权", "评价"],
    [
        ("欺诈", "0.851", "0.904", "0.94", "核心目标达成"),
        ("不违规", "0.985", "0.990", "0.99", "持平"),
        ("黑产广告", "0.950", "0.963", "0.96", "小幅提升"),
        ("博彩", "0.940", "0.941", "0.92", "持平"),
        ("谩骂引战", "0.934", "0.912", "0.87", "略有牺牲"),
        ("低俗色情", "0.920", "0.895", "0.94", "精确率下降（谩骂→低俗 24 条）"),
    ],
    [Inches(1.9), Inches(1.9), Inches(1.9), Inches(1.9), Inches(4.5)],
    size=12, row_h=Inches(0.42),
)
s = prs.slides[-1]
note_box(s, Inches(4.85),
         "加权只救回了「样本不均衡」这一部分：诊断发现约一半误判属于标注边界模糊"
         "（「上岸」「打米」「七位数」是博彩黑话，欺诈与博彩语料本身犬牙交错），继续调参收益有限。",
         h=Inches(0.72))
notes(s, "表格从左到右念，但只重点说三行：欺诈（涨）、谩骂（掉）、低俗（掉）。\n"
         "然后给一个'有洞察力'的结论：加权不是万能的——"
         "我做过误判样本分析，约一半是**标注边界模糊**，属于天花板问题，不是调参能解决的。\n"
         "这句会让评审觉得你不是只会调参，而是真的看过错例。")

# ================================================================ 13. 消融 2：蒸馏 + 剪枝
s = blank(prs)
header(s, "四、模型方案对比", "消融实验 ②：蒸馏 α 与剪枝稀疏度")
rect(s, M, Inches(1.66), Inches(5.95), Inches(1.1), fill=CARD)
textbox(s, M + Inches(0.2), Inches(1.78), Inches(5.6), Inches(0.9),
        "知识蒸馏 12 层 → 4 层（Teacher→Student）\n"
        "只抽层不蒸馏：macro-F1 仅 0.2445 → 蒸馏后 0.9230（+0.679）",
        size=11.5, spacing=1.35, color=INK)
gf = None
tbl_h = ["配置", "accuracy", "macro-F1", "欺诈召回"]
rows = [
    ("α = 0.5, 3 epochs", "0.9153", "0.9133", "0.77"),
    ("α = 0.7, 5 epochs（采用）", "0.9247", "0.9230", "0.83"),
]
gf = s.shapes.add_table(3, 4, M, Inches(2.9), Inches(5.95), Inches(1.4))
t = gf.table
for c, h in enumerate(tbl_h):
    t.cell(0, c).text = h
for r, row in enumerate(rows, start=1):
    for c, v in enumerate(row):
        t.cell(r, c).text = v
style_table(t, [Inches(2.35), Inches(1.2), Inches(1.2), Inches(1.2)], size=11)

rect(s, Inches(6.75), Inches(1.66), Inches(5.97), Inches(1.1), fill=RGBColor(0xFF, 0xF9, 0xEC))
textbox(s, Inches(6.95), Inches(1.78), Inches(5.6), Inches(0.9),
        "踩过的坑：α 别照抄论文的 0.5\n"
        "KL 软损失不带类别权重，占一半话语权就把少数类拉平了 —— 欺诈召回从 0.94 掉回 0.77。",
        size=11.5, spacing=1.35, color=AMBER)
t2 = s.shapes.add_table(4, 4, Inches(6.75), Inches(2.9), Inches(5.97), Inches(1.8)).table
for c, h in enumerate(["目标稀疏度", "实际稀疏度", "val_acc", "val_f1"]):
    t2.cell(0, c).text = h
for r, row in enumerate([("17%", "16.67%", "0.9407", "0.9414"),
                         ("33%", "33.33%", "0.9322", "0.9330"),
                         ("50%", "50.00%", "0.9396", "0.9390")], start=1):
    for c, v in enumerate(row):
        t2.cell(r, c).text = v
style_table(t2, [Inches(1.6), Inches(1.6), Inches(1.35), Inches(1.42)], size=11)
note_box(s, Inches(4.85),
         "剪枝的关键认知：未剪枝基线 val_f1 = 0.9441 → 剪掉一半权重只掉 0.5 个点，"
         "说明这个 12 层模型在广告审核任务上明显过参数化；但非结构化剪枝省的是 gzip 体积，不省 CPU 时间。",
         h=Inches(0.72))
notes(s, "这一页信息最密，建议按'左→右→底'讲：\n"
         "左：蒸馏。最震撼的一句话是'只抽层不蒸馏，macro-F1 只有 0.2445'——"
         "说明蒸馏不是可选项，而是让 Student 能用的唯一途径。\n"
         "α 消融讲'照抄论文会翻车'：KL 不带类别权重，会把上一步类别加权的收益弄丢，"
         "提到 0.7 才恢复。这是很硬的经验。\n"
         "右：剪枝稀疏度阶梯。结论是'剪一半只掉 0.5 个点'→ 模型冗余度高。"
         "但一定要补那句'非结构化剪枝不带来 CPU 加速'，否则会被认为不懂稀疏计算。")

# ================================================================ 14. 阈值拒判实验（图）
s = blank(prs)
header(s, "四、模型方案对比", "消融实验 ③：置信度阈值拒判（业务驱动的改进）")
cd = CategoryChartData()
cd.categories = ["0.00（纯 argmax）", "0.60（采用）", "0.80", "0.90"]
cd.add_series("自动判定占比", (100.0, 96.0, 90.9, 85.7))
cd.add_series("自动判定准确率", (93.10, 94.88, 96.55, 97.92))
cd.add_series("违规召回", (92.4, 90.2, 86.4, 82.1))
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, M, Inches(1.62), Inches(7.6), Inches(4.55), cd)
ch = gf.chart
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.BOTTOM
ch.legend.include_in_layout = False
ch.font.size = Pt(10.5)
ch.font.name = FONT
for ser in ch.plots[0].series:
    ser.smooth = False
ch.plots[0].has_data_labels = False
rect(s, Inches(8.45), Inches(1.62), Inches(4.27), Inches(4.55), fill=CARD)
textbox(s, Inches(8.65), Inches(1.8), Inches(3.9), Inches(4.2),
        "怎么读这张图：\n\n"
        "阈值 0.00 就是纯 argmax：\n全自动，但准确率只有 93.10%\n\n"
        "阈值 0.60（最终采用）：\n4% 流量转人工，自动判定准确率\n提升到 94.88%\n\n"
        "阈值 0.90：\n准确率最高 97.92%，但违规召回\n跌到 82.1% —— 漏放太多，不可接受\n\n"
        "结论：0.60 是「准确率提升」与\n「召回损失」的平衡点。",
        size=11.5, spacing=1.35, color=INK)
footer(s, "四个阈值点是实验报告里的实测工作点；前端工作台落在中间时按线性插值并标注「插值估计」")
notes(s, "这页是'业务驱动的改进'，一定要讲出权衡味道：\n"
         "核心句：'argmax 永远会挑一个类别，但审核业务需要「我不确定」这个答案。'\n"
         "读图顺序：从左到右，红线（违规召回）一直降、绿线（准确率）一直升，"
         "选 0.60 是因为再往上召回掉得太多，漏放违规的代价我们承担不起。\n"
         "如果被问'阈值怎么定的'：答：按业务能接受的违规召回下限反推，"
         "并用实测四个工作点做验证；真实上线应结合人工产能与真实先验重新标定。")

# ================================================================ 15. 最终模型架构
s = blank(prs)
header(s, "五、最终模型方案", "模型架构与推理契约")
rect(s, M, Inches(1.66), Inches(12.1), Inches(1.15), fill=CARD)
textbox(s, M + Inches(0.22), Inches(1.8), Inches(11.6), Inches(0.9),
        "输入文本 → BertTokenizer(max_len=64) → bert-base-chinese 12 层编码 → 取 [CLS] 向量 →"
        " Dropout → Linear(768 → 6) → softmax → 六类概率",
        size=13, spacing=1.35, color=INK)
rows = [
    ("训练设置", "epochs 3 · batch_size 8 · lr 2e-5 · weight_decay 1e-2 · 类别加权 CE"),
    ("存盘策略", "按最优验证 macro-F1 存盘（而非最后一轮）：验证 F1 在第 4 轮达峰 0.9257，第 5 轮回落到 0.9160"),
    ("拒判规则", "max(softmax) < conf_threshold(0.60) → 输出「需人工复核」，原判保留在 raw_class"),
    ("返回契约", "pred_class · raw_class · confidence · need_review · probs(六类概率数组)"),
    ("设备", "有 GPU 用 GPU 训练；量化与蒸馏推理强制 CPU（避免设备不一致导致的报错）"),
]
y = Inches(3.0)
for i, (k, v) in enumerate(rows):
    rect(s, M, y, Inches(12.1), Inches(0.7), fill=CARD if i % 2 == 0 else WHITE)
    textbox(s, M + Inches(0.22), y + Inches(0.17), Inches(1.9), Inches(0.4), k, size=12.5, bold=True, color=ACCENT)
    textbox(s, M + Inches(2.25), y + Inches(0.15), Inches(9.6), Inches(0.5), v, size=11.5, spacing=1.25)
    y += Inches(0.76)
notes(s, "这页讲'最终交付的模型长什么样'。\n"
         "重点两处：\n"
         "1) **按最优 F1 存盘而不是最后一轮**：我验证过 F1 在第 4 轮达峰、第 5 轮回落，"
         "如果按最后一轮存盘就白丢 1 个点。这是很实际的工程细节。\n"
         "2) 拒判规则与返回契约：这是下一节前端设计的接口基础（pred_class / confidence / need_review / probs）。")

# ================================================================ 16. 评测指标 + 混淆矩阵
s = blank(prs)
header(s, "五、最终模型方案", "评测指标与混淆矩阵")
kpis = [("0.9310", "accuracy"), ("0.9343", "macro-F1"), ("0.94", "欺诈召回"),
        ("390.2MB", "模型体积"), ("5.19ms", "模型前向"), ("96.0%", "自动判定占比")]
bw = Inches(1.92)
for i, (v, k) in enumerate(kpis):
    x = M + i * (bw + Inches(0.14))
    rect(s, x, Inches(1.62), bw, Inches(1.12), fill=CARD)
    textbox(s, x + Inches(0.1), Inches(1.76), bw - Inches(0.2), Inches(0.45), v, size=17, bold=True, color=ACCENT)
    textbox(s, x + Inches(0.1), Inches(2.22), bw - Inches(0.2), Inches(0.4), k, size=11, color=MUT)
tbl = s.shapes.add_table(7, 7, M, Inches(2.96), Inches(12.1), Inches(2.9)).table
head = ["真实 \\ 预测", "不违规", "黑产广告", "欺诈", "博彩", "谩骂引战", "低俗色情"]
for c, h in enumerate(head):
    tbl.cell(0, c).text = h
conf_data = [
    ("不违规 (100)", ("99", "0", "0", "0", "0", "1")),
    ("黑产广告 (188)", ("0", "181", "6", "0", "0", "1")),
    ("欺诈 (100)", ("0", "5", "94", "1", "0", "0")),
    ("博彩 (176)", ("0", "2", "7", "161", "0", "6")),
    ("谩骂引战 (191)", ("1", "0", "0", "0", "166", "24")),
    ("低俗色情 (201)", ("0", "0", "1", "4", "7", "189")),
]
for r, (name, vals) in enumerate(conf_data, start=1):
    tbl.cell(r, 0).text = name
    for c, v in enumerate(vals, start=1):
        tbl.cell(r, c).text = v
style_table(tbl, [Inches(2.3), Inches(1.63), Inches(1.63), Inches(1.63), Inches(1.63), Inches(1.63), Inches(1.65)],
            size=10.5, header_size=10.5)
footer(s, "混淆矩阵为测试集 956 条实测；最大混淆源：谩骂引战 → 低俗色情 24 条（辱骂文本常带性羞辱词，语义天然交叠）")
notes(s, "指标卡快速念，重点在混淆矩阵。\n"
         "要指出的两处：\n"
         "1) 对角线很干净，说明六类整体可分。\n"
         "2) **最刺眼的是谩骂引战 → 低俗色情 24 条**：辱骂里常带性羞辱词，"
         "两类语义天然交叠，这也是为什么低俗色情的精确率在加权后下降。\n"
         "如果被问'为什么欺诈只有 100 条测试样本'：答：数据集原始分布如此，"
         "这也是引入 macro-F1 而不是只看准确率的原因。")

# ================================================================ 17. 压缩方法
s = blank(prs)
header(s, "六、模型部署", "压缩方法：量化 / 蒸馏 / 剪枝")
cd = CategoryChartData()
cd.categories = ["原始模型", "动态量化 int8", "知识蒸馏 12→4 层", "剪枝 50%", "剪枝 + 量化"]
cd.add_series("存盘体积(MB)", (390.2, 145.6, 173.9, 390.2, 145.6))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, M, Inches(1.62), Inches(5.6), Inches(3.56), cd)
ch = gf.chart
ch.has_legend = False
ch.font.size = Pt(10)
ch.font.name = FONT
ch.plots[0].has_data_labels = True
ch.plots[0].data_labels.font.size = Pt(10)
ch.plots[0].gap_width = 60
ch.plots[0].series[0].format.fill.solid()
ch.plots[0].series[0].format.fill.fore_color.rgb = ACCENT

tbl = s.shapes.add_table(6, 4, Inches(6.45), Inches(1.62), Inches(6.27), Inches(3.6)).table
for c, h in enumerate(["手段", "体积", "accuracy", "macro-F1"]):
    tbl.cell(0, c).text = h
rows = [
    ("原始模型（加权）", "390.2 MB", "0.9310", "0.9343"),
    ("动态量化 int8", "145.6 MB", "0.9150", "0.9150"),
    ("知识蒸馏（4 层）", "173.9 MB", "0.9247", "0.9230"),
    ("非结构化剪枝 50%", "390.2 MB", "0.9320", "0.9305"),
    ("剪枝 + 量化", "145.6 MB", "0.9163", "0.9146"),
]
for r, row in enumerate(rows, start=1):
    for c, v in enumerate(row):
        tbl.cell(r, c).text = v
style_table(tbl, [Inches(2.0), Inches(1.35), Inches(1.35), Inches(1.57)], size=10.5)
note_box(s, Inches(5.42),
         "三条结论：① 量化最省体积（390→146MB，CPU 1.41×）；② 蒸馏最省延迟（1.68ms，3.08×）；"
         "③ 剪枝省的是 gzip 体积（244.3MB）不是磁盘也不是速度 —— 稀疏矩阵需要专用算子才能加速。",
         h=Inches(0.86))
notes(s, "这一页的核心是'三种手段各自解决不同问题'，不要说成'都能压缩'。\n"
         "量化 → 体积（CPU 上 1.41× 加速，精度掉 1.8 个点）\n"
         "蒸馏 → 延迟与参数量（45.6M 参数、3.08× 加速）\n"
         "剪枝 → 只省 gzip 体积，**不带来 CPU 加速**（这是最容易被误答的点）\n"
         "补充：我是按'量化版粗筛 + 原始版复核'的分层部署思路来选的，不是单点最优。")

# ================================================================ 18. 接口设计
s = blank(prs)
header(s, "六、模型部署", "部署方案 ① 接口设计")
rect(s, M, Inches(1.62), Inches(12.1), Inches(0.86), fill=CARD)
textbox(s, M + Inches(0.22), Inches(1.74), Inches(11.6), Inches(0.65),
        "四条路线统一契约：POST /predict，请求 {\"text\": \"…\"}，"
        "返回 pred_class · raw_class · confidence · need_review · probs（大模型为自报分数）",
        size=12.5, color=INK, spacing=1.3)
tbl = s.shapes.add_table(5, 5, M, Inches(2.66), Inches(12.1), Inches(2.3)).table
for c, h in enumerate(["路线", "服务端口", "置信度来源", "拒判", "状态"]):
    tbl.cell(0, c).text = h
rows = [
    ("BERT 微调（主）", "8000", "softmax", "✓", "已部署"),
    ("随机森林 + TF-IDF", "8001", "200 棵树投票比例", "✓", "已部署"),
    ("fastText（字级）", "8002", "6 类归一化得分", "✓", "已部署"),
    ("大模型 DeepSeek", "5002", "模型自报分数（未校准）", "✓", "已部署"),
]
for r, row in enumerate(rows, start=1):
    for c, v in enumerate(row):
        tbl.cell(r, c).text = v
style_table(tbl, [Inches(3.1), Inches(1.5), Inches(4.0), Inches(1.2), Inches(2.3)], size=11)
note_box(s, Inches(5.12),
         "一个必须知道的坑：端口**不能用 6000**。6000 在 Chromium（Chrome/Edge）的受限端口名单里，"
         "浏览器会直接拒绝连接并报 ERR_UNSAFE_PORT，而 curl / requests 不检查该名单 —— "
         "命令行测试一切正常，只有网页端连不上。",
         color=AMBER, fill=RGBColor(0xFF, 0xF9, 0xEC), h=Inches(0.86))
notes(s, "接口设计讲三点：\n"
         "1) **统一契约**：四条路线字段一致（大模型的分数是自报，要标注清楚），"
         "所以前端一次适配就能切换。\n"
         "2) **端口重新分配**：原来是三条路线都抢 6000，既是冲突、又踩了浏览器的受限端口。"
         "现在 8000/8001/8002/5002 互不冲突、都可被浏览器访问。\n"
         "3) ERR_UNSAFE_PORT 这个坑值得讲——它很难查，因为 curl 永远正常。"
         "（说这句会显得你真踩过坑而不是照抄教程。）")

# ================================================================ 19. 前端设计
s = blank(prs)
header(s, "六、模型部署", "部署方案 ② 前端设计：单文件审核工作台")
# 布局示意图
rect(s, M, Inches(1.66), Inches(7.6), Inches(4.5), fill=WHITE, line=LINE)
rect(s, M, Inches(1.66), Inches(7.6), Inches(0.44), fill=ACCENT)
textbox(s, M + Inches(0.16), Inches(1.72), Inches(7.3), Inches(0.32),
        "顶部状态条：模型路线切换 · 置信度阈值 · 接口状态 · 设置", size=10.5, color=WHITE)
rect(s, M, Inches(2.1), Inches(1.55), Inches(4.06), fill=CARD)
for i, t in enumerate(["审核工作台", "批量审核", "审核记录", "指标看板", "方案对比", "接口与部署"]):
    textbox(s, M + Inches(0.1), Inches(2.34 + i * 0.6), Inches(1.4), Inches(0.4), t, size=9.5,
            color=ACCENT if i == 0 else MUT, bold=(i == 0))
rect(s, M + Inches(1.7), Inches(2.24), Inches(5.75), Inches(2.2), fill=CARD)
textbox(s, M + Inches(1.86), Inches(2.38), Inches(5.4), Inches(1.95),
        "主工作区：输入文案 → 开始审核\n\n"
        "结论卡：判定类别大字 + 置信度 + 处置徽章\n"
        "（自动通过 / 自动拦截 / 转人工复核）\n\n"
        "六类概率条（真实分布，按概率降序，前 3 高亮）",
        size=10.5, color=INK, spacing=1.35)
rect(s, M + Inches(1.7), Inches(4.56), Inches(5.75), Inches(1.6), fill=CARD)
textbox(s, M + Inches(1.86), Inches(4.68), Inches(5.4), Inches(1.4),
        "右栏：处置分流（阈值-准确率-召回三方权衡，取实验实测工作点）\n"
        "右栏：最近审核记录（点击可回填重审）",
        size=10.5, color=INK, spacing=1.35)
textbox(s, Inches(8.55), Inches(1.72), Inches(4.18), Inches(4.4),
        "设计要点\n\n"
        "① 单文件 HTML，双击即用：无构建、无依赖、无 CDN，断网可用\n\n"
        "② 六区工作台取代原来 4 个独立 Streamlit 演示页\n\n"
        "③ 离线兜底：接口不可用时用内置规则演示，并在结论上明确标注「非模型推理」，不伪装\n\n"
        "④ 同源托管：服务根路径直接托管页面，前端与接口同源，彻底绕开跨域\n\n"
        "⑤ 每条记录都带「真实接口 / 内置规则」来源与模型路线，便于追溯",
        size=10.5, color=INK, spacing=1.4)
footer(s, "现场可直接演示：切换四条路线 → 同一批文案的判定与置信度差异一目了然")
notes(s, "前端这页最好现场演示，讲不出效果就照这三句讲：\n"
         "1) **取代了 4 个分散的 Streamlit 页**，现在一个工作台里能横向对比四条路线。\n"
         "2) **离线兜底 + 明确标注**：这是有意设计的诚实性——"
         "接口挂了就用内置规则，但结论上会写「非模型推理」，绝不让人误以为是模型输出。\n"
         "3) **同源托管**：把前端挂在服务根路径，前端与接口同源，"
         "从根上避开跨域；另外端口必须避开 6000（上一个坑）。\n"
         "演示脚本：打开 8000 的工作台 → 输入「蠢猪」→ BERT 给谩骂引战 98.7% → "
         "切到随机森林 → 同一句给出低俗色情 0.42 且转人工。这个对比很有冲击力。")

# ================================================================ 20. 总结与局限
s = blank(prs)
header(s, "总结", "结论 · 局限 · 口径提醒")
rect(s, M, Inches(1.62), Inches(5.95), Inches(2.2), fill=ACCENT_L)
textbox(s, M + Inches(0.2), Inches(1.76), Inches(5.6), Inches(1.95),
        "三个结论\n"
        "① BERT 加权（0.9310 / 0.9343）显著优于传统基线，是最终采用的主路线；\n"
        "② 拒判让 96% 流量自动化、自动判定准确率 +1.8 个点，人工产能放大约 25 倍；\n"
        "③ 量化省体积、蒸馏省延迟、剪枝只省 gzip 体积 —— 按场景组合，而非单点最优。",
        size=11.5, color=ACCENT, spacing=1.45)
rect(s, Inches(6.75), Inches(1.62), Inches(5.97), Inches(2.2), fill=CARD)
textbox(s, Inches(6.95), Inches(1.76), Inches(5.6), Inches(1.95),
        "四条局限（教学项目，不能直接上生产）\n"
        "① 先验分布不同：实验集违规近 90%，真实投放远低于此，阈值须重新标定；\n"
        "② 仅覆盖文本：真实素材含图片/视频，图文不一致是常见违规手法；\n"
        "③ 标签体系不全：六类覆盖不了医疗器械夸大、金融保本承诺等专项条款；\n"
        "④ 黑话漂移：黑产话术迭代快，模型需要持续用新样本运营更新。",
        size=11.5, color=INK, spacing=1.4)
rect(s, M, Inches(4.0), Inches(12.1), Inches(2.1), fill=RGBColor(0xFF, 0xF9, 0xEC))
textbox(s, M + Inches(0.22), Inches(4.14), Inches(11.6), Inches(1.85),
        "口径提醒（汇报与文档里最容易搞错的四处）\n"
        "① README 的「准确率 0.9331」是错的：0.9331 是 dev 验证集的 val_f1（macro-F1），不是准确率；"
        "准确率口径应为 0.9310，macro-F1 为 0.9343。\n"
        "② 随机森林用的是自带 8:2 切分，不是统一测试集，不能与其它方案直接横向比较。\n"
        "③ 延迟有三个口径：模型前向 5.19ms / CPU 端到端 103ms / 剪枝表 63.6ms，引用时必须说明取自哪张表。\n"
        "④ 四条路线的「置信度」不是一回事：softmax、投票比例、归一化得分、模型自报分数，数值不可横向比较。",
        size=11, color=AMBER, spacing=1.35)
bottom = rect(s, M, Inches(6.24), Inches(12.1), Inches(0.6), fill=ACCENT)
textbox(s, M + Inches(0.2), Inches(6.34), Inches(11.7), Inches(0.42),
        "感谢聆听 · 欢迎提问      模型输出仅供参考，不构成法律意见；实际投放以监管要求与法务审核为准。",
        size=12, color=WHITE, bold=True)
notes(s, "收尾三件事，按顺序：\n"
         "1) 三个结论：选 BERT 的理由、拒判的业务价值、三种压缩手段的分工。\n"
         "2) **主动说局限**——这比被问出来强得多。尤其第①条'先验分布不同'，"
         "是这类项目最本质的问题。\n"
         "3) **主动交代口径错误**：README 里那个 0.9331 是 val_f1 不是准确率，"
         "我已经在报告和前端里纠正为 0.9310 / 0.9343。"
         "主动认这一点会显著加分，也会让老师相信其它数字都核对过。\n"
         "结尾：如果被问到答不上来的，就说'这个我还没验证过，回去补'——不要编。")

prs.save(OUT)
print("已生成：%s" % OUT)
print("共 %d 页" % len(prs.slides.__iter__.__self__._sldIdLst))
