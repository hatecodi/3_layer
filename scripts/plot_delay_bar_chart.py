# -*- coding: utf-8 -*-
"""
受天气影响车辆的平均延误时间 —— 分组柱状图（SCI 论文风格）
数据来源：D:\\图\\受天气影响车辆的平均延误时间.xlsx
依赖：matplotlib、numpy（缺失时先 pip install matplotlib numpy）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ===================== 0. 中文字体注册（彻底解决中文乱码与负号异常） =====================
ZH_FONT_PATH = r"C:\Windows\Fonts\simsun.ttc"          # 宋体：衬线，同时含中英文
font_manager.fontManager.addfont(ZH_FONT_PATH)
zh_font_name = font_manager.FontProperties(fname=ZH_FONT_PATH).get_name()

plt.rcParams.update({
    # 直接指定字体名列表：英文/数字用 Times New Roman，中文缺字自动回退到宋体
    "font.family": ["Times New Roman", zh_font_name],
    "mathtext.fontset": "stix",                        # 数学符号与 Times 风格一致
    "axes.unicode_minus": False,                       # 正常显示负号
    "axes.linewidth": 1.2,                             # 坐标轴线宽 1.2 pt
    "font.size": 10.5,                                 # 基础字号
})

# ===================== 1. 数据 =====================
x_labels = ["10_t", "15_t", "20_t", "25_t", "30_t"]      # 输入规模，共 5 组
methods = ["HMAPPO", "MAPPO", "SA", "SA-nofc", "调度员规则"]

data = {
    "HMAPPO":     [34.2,  38.5, 100.4, 113.9, 119.7],
    "MAPPO":      [74.7,  81.9, 162.8, 174.1, 178.9],
    "SA":         [30.5,  30.1, 116.7, 114.6, 124.9],
    "SA-nofc":    [79.5,  80.9, 172.1, 180.4, 185.9],
    "调度员规则":  [87.0,  88.4, 174.0, 181.2, 185.6],
}

# 高对比度学术配色
colors = {
    "HMAPPO":     "#4C72B0",   # 蓝
    "MAPPO":      "#DD8452",   # 橙
    "SA":         "#55A868",   # 绿
    "SA-nofc":    "#C44E52",   # 红
    "调度员规则":  "#8172B3",   # 紫
}

# 差异化填充纹理，确保黑白印刷时各组仍可清晰区分
# hatches = {
#     "HMAPPO":     "///",     # 斜线
#     "MAPPO":      "xxx",     # 交叉
#     "SA":         "---",     # 横线
#     "SA-nofc":    "+++",     # 加号
#     "调度员规则":  "ooo",     # 圆圈
# }

# ===================== 2. 绘制分组柱状图 =====================
n_groups = len(x_labels)
n_methods = len(methods)
bar_width = 0.15                                      # 单根柱子宽度
x = np.arange(n_groups)                               # 各组中心位置

fig, ax = plt.subplots(figsize=(6.5, 4.2))

for i, method in enumerate(methods):
    offset = (i - (n_methods - 1) / 2) * bar_width    # 组内居中排布
    ax.bar(
        x + offset,
        data[method],
        width=bar_width,
        label=method,
        color=colors[method],
        # hatch=hatches[method],
        edgecolor="black",    # 黑色描边，纹理也为黑色，便于黑白印刷
        linewidth=0.4,
    )

# ===================== 3. 坐标轴、标题与刻度 =====================
ax.set_xlabel("输入规模", fontsize=11.5)
ax.set_ylabel("平均延误时间", fontsize=11.5)
ax.set_title("受天气影响车辆的平均延误时间", fontsize=13, pad=10)  # 标题 > 轴标签 > 刻度

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=10)
ax.tick_params(axis="both", direction="in", length=4, labelsize=10)  # 刻度线向内

# ===================== 4. 网格 =====================
# 浅色横向虚线网格，透明度低，置于柱子后方不干扰展示
ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="gray")
ax.set_axisbelow(True)

# 去掉上、右边框（SCI 期刊常用简洁风格）
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

# ===================== 5. 图例（无边框，位置不遮挡数据主体） =====================
ax.legend(frameon=False, loc="upper left", fontsize=9.5)

# ===================== 6. 布局与保存 =====================
fig.tight_layout()  # 自动调整布局，避免标签被截断

# 位图输出：600 dpi（满足 ≥300 dpi 投稿要求）
fig.savefig(r"D:\图\受天气影响车辆的平均延误时间.png", dpi=600, bbox_inches="tight")

# 矢量图输出：PDF（SCI 投稿推荐，取消下方注释即可生成）
# fig.savefig(r"D:\图\受天气影响车辆的平均延误时间.pdf", bbox_inches="tight")

plt.show()
