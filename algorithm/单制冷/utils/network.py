import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arrow, FancyArrowPatch

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

def draw_residual_rnn():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-5, 15)
    ax.set_ylim(-3, 8)
    ax.axis('off')

    # 颜色定义
    input_color = '#a8d1ff'       # 输入特征
    rnn_color = '#ffd966'         # RNN层
    fc_color = '#b6e388'          # 全连接层fc_V
    residual_color = '#ffaaa5'    # 残差连接
    arrow_color = '#555555'       # 箭头

    # -------------------------- 单时间步t的结构 --------------------------
    # 1. 输入层（分迭代特征和固定特征）
    # 迭代特征输入（itera_dim=7）
    ax.add_patch(Rectangle((-4, 5), 2, 1.5, fc=input_color, ec='black', lw=2))
    ax.text(-3, 5.75, '迭代特征 x_itera(t)\n(7维)', ha='center', va='center', fontsize=10)
    # 固定特征输入（fixed_dim=5）
    ax.add_patch(Rectangle((-4, 3), 2, 1.5, fc=input_color, ec='black', lw=2, hatch='/'))
    ax.text(-3, 3.75, '固定特征 x_fixed\n(5维)', ha='center', va='center', fontsize=10)

    # 2. RNN层（处理迭代特征）
    ax.add_patch(Rectangle((0, 4), 3, 3, fc=rnn_color, ec='black', lw=2))
    ax.text(1.5, 5.5, 'RNN层\n(h_dim=2048)', ha='center', va='center', fontsize=11, fontweight='bold')

    # 3. 隐藏状态h(t)
    ax.add_patch(Circle((4.5, 5.5), 0.5, fc=rnn_color, ec='black', lw=2))
    ax.text(4.5, 5.5, 'h(t)', ha='center', va='center', fontsize=10)

    # 4. 拼接操作（h(t) + x_fixed）
    ax.add_patch(Rectangle((5.5, 4), 2, 3, fc='white', ec='black', lw=2, linestyle='--'))
    ax.text(6.5, 5.5, '拼接\n(h(t) + x_fixed)', ha='center', va='center', fontsize=10)

    # 5. 全连接网络fc_V（预测difft）
    ax.add_patch(Rectangle((8, 4), 3, 3, fc=fc_color, ec='black', lw=2))
    ax.text(9.5, 5.5, 'fc_V\n(残差预测)', ha='center', va='center', fontsize=11, fontweight='bold')

    # 6. 输出difft(t)（迭代特征变化量）
    ax.add_patch(Circle((12, 5.5), 0.5, fc=residual_color, ec='black', lw=2))
    ax.text(12, 5.5, 'difft(t)\n(7维)', ha='center', va='center', fontsize=10)

    # -------------------------- 时间步循环（t -> t+1） --------------------------
    # 7. 残差计算（x_itera(t+1) = x_itera(t) + difft(t)）
    ax.add_patch(Rectangle((-4, 1), 3, 1.5, fc=residual_color, ec='black', lw=2))
    ax.text(-2.5, 1.75, 'x_itera(t+1)\n= x_itera(t) + difft(t)', ha='center', va='center', fontsize=10)

    # 8. 下一时刻迭代特征x_itera(t+1)
    ax.add_patch(Circle((-4, -1), 0.5, fc=input_color, ec='black', lw=2))
    ax.text(-4, -1, 'x_itera(t+1)', ha='center', va='center', fontsize=10)

    # 9. 下一时刻隐藏状态h(t+1)
    ax.add_patch(Circle((4.5, -1), 0.5, fc=rnn_color, ec='black', lw=2))
    ax.text(4.5, -1, 'h(t+1)', ha='center', va='center', fontsize=10)

    # -------------------------- 箭头连接（数据流向） --------------------------
    # 输入到RNN
    arrow1 = FancyArrowPatch((-2, 5.75), (0, 5.75), arrowstyle='->', color=arrow_color, lw=2)
    ax.add_patch(arrow1)
    # 固定特征到拼接层
    arrow2 = FancyArrowPatch((-2, 3.75), (5.5, 3.75), arrowstyle='->', color=arrow_color, lw=2)
    ax.add_patch(arrow2)
    # RNN到h(t)
    arrow3 = FancyArrowPatch((3, 5.5), (4, 5.5), arrowstyle='->', color=arrow_color, lw=2)
    ax.add_patch(arrow3)
    # h(t)到拼接层
    arrow4 = FancyArrowPatch((5, 5.5), (5.5, 5.5), arrowstyle='->', color=arrow_color, lw=2)
    ax.add_patch(arrow4)
    # 拼接层到fc_V
    arrow5 = FancyArrowPatch((7.5, 5.5), (8, 5.5), arrowstyle='->', color=arrow_color, lw=2)
    ax.add_patch(arrow5)
    # fc_V到difft(t)
    arrow6 = FancyArrowPatch((11, 5.5), (11.5, 5.5), arrowstyle='->', color=arrow_color, lw=2)
    ax.add_patch(arrow6)

    # 时间步循环箭头
    # difft(t)到残差计算
    arrow7 = FancyArrowPatch((12.5, 5.5), (12.5, 2.25), arrowstyle='->', color=arrow_color, lw=2)
    ax.add_patch(arrow7)
    arrow8 = FancyArrowPatch((12.5, 2.25), (-2.5, 2.25), arrowstyle='->', color=arrow_color, lw=2)
    ax.add_patch(arrow8)
    # 迭代特征到残差计算
    arrow9 = FancyArrowPatch((-3, 5), (-3, 2.5), arrowstyle='->', color=arrow_color, lw=2)
    ax.add_patch(arrow9)
    # 残差计算到x_itera(t+1)
    arrow10 = FancyArrowPatch((-2.5, 1), (-3.5, 1), arrowstyle='->', color=arrow_color, lw=2)
    ax.add_patch(arrow10)
    arrow11 = FancyArrowPatch((-3.5, 1), (-3.5, -0.5), arrowstyle='->', color=arrow_color, lw=2)
    ax.add_patch(arrow11)
    # x_itera(t+1)到RNN（循环）
    arrow12 = FancyArrowPatch((-4, -1.5), (-4, 4), arrowstyle='->', color=arrow_color, lw=2, linestyle='--')
    ax.add_patch(arrow12)
    # RNN到h(t+1)（循环）
    arrow13 = FancyArrowPatch((1.5, 4), (1.5, -1), arrowstyle='->', color=arrow_color, lw=2, linestyle='--')
    ax.add_patch(arrow13)

    # 标题
    plt.title('residual_rnn循环神经网络结构', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

# 绘制并显示
fig = draw_residual_rnn()
plt.show()