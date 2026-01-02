import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# 使用 Noto Serif CJK JP
font_path = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
font_manager. fontManager.addfont(font_path)
plt.rcParams['font.sans-serif'] = ['Noto Serif CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据
categories = ['不下修公告', '提议下修公告']
has_prompt = [1738, 328]
no_prompt = [202, 13]
total = [1940, 341]

# 创建图表
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 图1: 分组柱状图
x = np. arange(len(categories))
width = 0.25

ax1 = axes[0]
bars1 = ax1.bar(x - width, has_prompt, width, label='有提示性公告', color='#5470c6')
bars2 = ax1.bar(x, no_prompt, width, label='无提示性公告', color='#91cc75')
bars3 = ax1.bar(x + width, total, width, label='总数', color='#fac858')

ax1.set_xlabel('公告类型', fontsize=12)
ax1.set_ylabel('数量', fontsize=12)
ax1.set_title('公告类型统计对比', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 在柱子上添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

# 图2: 堆叠柱状图
ax2 = axes[1]
bars1 = ax2.bar(categories, has_prompt, label='有提示性公告', color='#5470c6')
bars2 = ax2.bar(categories, no_prompt, bottom=has_prompt, label='无提示性公告', color='#91cc75')

ax2.set_xlabel('公告类型', fontsize=12)
ax2.set_ylabel('数量', fontsize=12)
ax2.set_title('公告类型堆叠统计', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 添加总数标签
for i, (cat, tot) in enumerate(zip(categories, total)):
    ax2.text(i, tot, f'总计: {tot}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('announcement_statistics.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n图表已保存为 'announcement_statistics. png'")

# 打印数据摘要
print("\n数据摘要:")
print(f"总公告数:  {sum(total)}")
print(f"有提示性公告总数: {sum(has_prompt)} ({sum(has_prompt)/sum(total)*100:.1f}%)")
print(f"无提示性公告总数: {sum(no_prompt)} ({sum(no_prompt)/sum(total)*100:.1f}%)")