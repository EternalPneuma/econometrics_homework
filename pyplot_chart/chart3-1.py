import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

times = font_manager.FontProperties(family='Times New Roman')
simsun = font_manager.FontProperties(family='SimSun')

# 数据
trading_days = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
cumulative_returns = [0.00, -0.13, -0.20, -0.24, -0.29, -0.28, -0.06, -0.02, 0.09, 0.25, 
                      0.63, 3.03, 3.14, 3.32, 3.42, 3.57, 3.58, 3.77, 4.02, 4.35, 4.72]

# 创建图表
fig, ax = plt.subplots(figsize=(12, 7))

# 添加阴影区域区分前后（放在最前面，作为背景）
ax.axvspan(-10, 0, alpha=0.1, color='blue', label='公告前')
ax.axvspan(0, 10, alpha=0.1, color='orange', label='公告后')

# 绘制折线图
line = ax.plot(trading_days, cumulative_returns, linewidth=2.5, color='#5470c6', 
               marker='o', markersize=6, markerfacecolor='white', markeredgewidth=2, 
               markeredgecolor='#5470c6', label='累计涨跌幅')

# 添加零线
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# 添加事件日（T=0）的垂直线
ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='提议下修公告日')

# 在事件日添加标注
ax.annotate('提议下修\n公告日', xy=(0, 0.63), xytext=(1.5, 0),
            fontsize=11, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

# 最高点
max_idx = cumulative_returns.index(max(cumulative_returns))
ax.plot(trading_days[max_idx], cumulative_returns[max_idx], 'ro', markersize=10, 
        markerfacecolor='lightcoral', markeredgewidth=2, markeredgecolor='red')
ax.annotate(f'{cumulative_returns[max_idx]:.2f}%',
            xy=(trading_days[max_idx], cumulative_returns[max_idx]),
            xytext=(trading_days[max_idx]-0.3, cumulative_returns[max_idx]+0.2),
            fontsize=10, color='red', fontweight='bold')

# 设置标题和标签
ax.set_xlabel('交易日（相对提议下修公告日）', fontsize=13)
ax.set_ylabel('累计涨跌幅（%）', fontsize=13)

# 设置x轴刻度
ax.set_xticks(trading_days)
ax.set_xticklabels([f'T{d:+d}' if d != 0 else 'T=0' for d in trading_days], rotation=45, ha='right')

# 添加网格
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# 设置y轴格式
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))

# 添加图例
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

# 添加文本说明
textstr = f'公告前累计收益:  {cumulative_returns[10]:.2f}%\n公告后累计收益:  {cumulative_returns[-1] - cumulative_returns[10]:.2f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.01, 0.79, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

# 留出上下边距，避免标注与边框/坐标轴挤在一起
ax.set_ylim(min(cumulative_returns) - 0.8, max(cumulative_returns) + 0.8)

# 使用 tight_layout
plt.tight_layout()
plt.savefig('cumulative_return_around_announcement.png', dpi=300, bbox_inches='tight')
plt.show()

print("图表已保存为 'cumulative_return_around_announcement.png'")

# 打印数据摘要
print("\n数据摘要:")
print(f"公告日收益:  T=0, {cumulative_returns[10]:.2f}%")
print(f"公告后最高点: T{trading_days[max_idx]}, {cumulative_returns[max_idx]:.2f}%")
print(f"事件窗口总收益: {cumulative_returns[-1] - cumulative_returns[0]:.2f}%")
print(f"公告后10日涨幅: {cumulative_returns[-1] - cumulative_returns[10]:.2f}%")