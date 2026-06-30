import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from datetime import datetime
from io import BytesIO
from PIL import Image

# ============================================================
# 1. 模型参数（与列线图代码完全一致）
# ============================================================
intercept = -1.859
coef = {
    "f/tPSA": -0.368,
    "fPSA": 0.377,
    "CCLmax": 1.054,
    "Capsular bulging": 1.095,
    "Capsular disruption": 1.107,
    "Capsular retraction": 0.701
}

# 对数转换所需的均值和标准差（来自训练集）
mu_fpsa = 0.6659
sigma_fpsa = 0.9259
mu_ftpsa = -2.0339
sigma_ftpsa = 0.7558
mu_cc = 34.566
sigma_cc = 35.530

# 各变量的原始值范围（2.5% ~ 97.5% 分位数）
ranges = {
    "f/tPSA": (0.04, 0.77),
    "fPSA": (0.49, 21.88),
    "CCLmax": (0.0, 105.0),
    "Capsular bulging": (0, 1),
    "Capsular disruption": (0, 1),
    "Capsular retraction": (0, 1)
}
binary_vars = ["Capsular bulging", "Capsular disruption", "Capsular retraction"]

# ===== 修改点 1：变量显示名称（列线图轴标签） =====
var_display_names = {
    "f/tPSA": "Free/Total PSA",
    "fPSA": "Free PSA",
    "CCLmax": "CCLmax",
    "Capsular bulging": "Capsular Bulging",
    "Capsular disruption": "Capsular Disruption",
    "Capsular retraction": "Capsular Retraction"
}

# ----- 计算列线图分数体系（与列线图代码完全一致） -----
def get_lp(var, x_raw):
    """计算线性预测值（不含截距）"""
    if var == "CCLmax":
        return coef[var] * (x_raw - mu_cc) / sigma_cc
    elif var == "fPSA":
        x_log = np.log(x_raw) if x_raw > 0 else np.log(1e-6)
        return coef[var] * (x_log - mu_fpsa) / sigma_fpsa
    elif var == "f/tPSA":
        x_log = np.log(x_raw) if x_raw > 0 else np.log(1e-6)
        return coef[var] * (x_log - mu_ftpsa) / sigma_ftpsa
    else:
        return coef[var] * x_raw

# 基线值：f/tPSA取最大值（反转），其余取最小值
baseline_values = {}
for var in coef:
    if var == "f/tPSA":
        baseline_values[var] = ranges[var][1]
    else:
        baseline_values[var] = ranges[var][0]

# 计算 scale 使得基线到最大风险对应 100 分（如需要可扩展）
lp_all_max = sum(get_lp(v, ranges[v][1]) for v in coef)
baseline_lp = sum(get_lp(v, baseline_values[v]) for v in coef)
logit_baseline = baseline_lp + intercept
logit_max = lp_all_max + intercept
max_prob = expit(logit_max)

TARGET_PROB = 0.9
if max_prob < TARGET_PROB:
    target_logit = np.log(TARGET_PROB / (1 - TARGET_PROB))
    extra_logit = target_logit - logit_max
    scale = 100 / (lp_all_max - baseline_lp)
    MAX_POINTS = 100 + extra_logit * scale
else:
    scale = 100 / (lp_all_max - baseline_lp)
    MAX_POINTS = 100
MAX_POINTS = int(np.ceil(MAX_POINTS / 10) * 10)   # 取整到10的倍数

def calc_points(var, x_raw):
    """计算单个变量的得分（点数）"""
    base_lp = get_lp(var, baseline_values[var])
    return (get_lp(var, x_raw) - base_lp) * scale

def inv_calc_points(var, points):
    """根据得分反推原始值（用于绘制刻度）"""
    base_lp = get_lp(var, baseline_values[var])
    lp_val = points / scale + base_lp
    if var == "CCLmax":
        return lp_val / coef[var] * sigma_cc + mu_cc
    elif var == "fPSA":
        log_x = lp_val / coef[var] * sigma_fpsa + mu_fpsa
        return np.exp(log_x)
    elif var == "f/tPSA":
        log_x = lp_val / coef[var] * sigma_ftpsa + mu_ftpsa
        return np.exp(log_x)
    else:
        return None

# ============================================================
# 2. 绘图函数：列线图（含红点标记）
# ============================================================
def plot_nomogram(case):
    """
    绘制列线图，并在每个变量轴、总分轴、风险轴标记红点
    case: dict, 包含所有变量的值
    返回 matplotlib.figure.Figure
    """
    # 绘图参数（与列线图代码一致）
    figsize = (12, 11)
    left_margin = 50   # 适当增加左侧空白，避免标签重叠
    axis_gap = 1.0
    y_points = 9.5
    y_ftpsa  = y_points - axis_gap
    y_fpsa   = y_ftpsa  - axis_gap
    y_cc     = y_fpsa   - axis_gap
    y_bulge  = y_cc     - axis_gap
    y_disrupt = y_bulge - axis_gap
    y_retract = y_disrupt - axis_gap
    y_total  = y_retract - axis_gap
    y_prob   = y_total  - axis_gap

    font_family = 'Arial'
    label_fontsize = 18
    tick_fontsize = 14
    text_color = 'black'
    line_color = 'black'
    line_width = 1.5
    tick_length = 0.15
    tick_width = 1.0
    label_offset = 0.25

    points_tick_direction = 'down'
    others_tick_direction = 'up'

    variables = list(coef.keys())
    var_y = {
        "f/tPSA": y_ftpsa,
        "fPSA": y_fpsa,
        "CCLmax": y_cc,
        "Capsular bulging": y_bulge,
        "Capsular disruption": y_disrupt,
        "Capsular retraction": y_retract
    }

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-left_margin, MAX_POINTS + 10)
    ax.set_ylim(-0.5, y_points + 1.5)
    ax.axis('off')

    # ---- 辅助函数：获取刻度偏移 ----
    def get_tick_offsets(direction):
        if direction == 'up':
            return tick_length, -label_offset
        else:
            return -tick_length, label_offset

    # ---- 绘制每个变量轴 ----
    for var in variables:
        y = var_y[var]
        xmin, xmax = ranges[var]

        # 计算该变量轴的刻度位置（points值）和对应的原始值标签
        if var in binary_vars:
            ticks = np.array([0, 1])
            tick_points = np.array([calc_points(var, t) for t in ticks])
        else:
            pt_min = calc_points(var, xmin)
            pt_max = calc_points(var, xmax)
            pt_low, pt_high = min(pt_min, pt_max), max(pt_min, pt_max)
            pt_vals = np.linspace(pt_low, pt_high, 4)
            raw_vals = np.array([inv_calc_points(var, p) for p in pt_vals])
            raw_vals = np.clip(raw_vals, min(xmin, xmax), max(xmin, xmax))
            ticks = raw_vals
            tick_points = pt_vals

        # 绘制轴线
        ax.plot([tick_points.min(), tick_points.max()], [y, y],
                color=line_color, lw=line_width)

        tick_shift, label_shift = get_tick_offsets(others_tick_direction)

        # 绘制刻度线和标签
        for t, tp in zip(ticks, tick_points):
            ax.plot([tp, tp], [y, y + tick_shift], color=line_color, lw=tick_width)
            if var in binary_vars:
                label = f"{int(t)}"
            else:
                if np.isclose(t, 0.0, atol=1e-6):
                    label = "0"
                else:
                    label = f"{t:.1f}"
            va = 'bottom' if label_shift > 0 else 'top'
            ax.text(tp, y + label_shift, label, ha='center', va=va,
                    fontsize=tick_fontsize, color=text_color)

        # 变量名（使用新的显示名称）
        display_name = var_display_names.get(var, var)
        ax.text(-left_margin + 5, y, display_name, ha='left', va='center',
                fontsize=label_fontsize, color=text_color, fontweight='bold')

        # ---- 在变量轴上标记红点（当前病例） ----
        raw_val = case[var]
        # 处理二分类：将bool转为0/1
        if var in binary_vars:
            raw_val = 1 if raw_val else 0
        # 计算该变量的得分
        score = calc_points(var, raw_val)
        # 如果得分超出轴范围，截断到端点（但红点仍显示在端点）
        # 找到轴上的实际范围
        ax_range_min = tick_points.min()
        ax_range_max = tick_points.max()
        # 红点位置
        x_red = np.clip(score, ax_range_min, ax_range_max)
        ax.plot(x_red, y, 'ro', markersize=10, markeredgecolor='darkred', zorder=5)
        # 显示得分数值（可选）
        ax.text(x_red - 2, y + 0.2, f'{score:.1f}', fontsize=12, color='red',
                ha='right', va='bottom', fontweight='bold')

    # ---- Points 轴（最上方） ----
    ax.plot([0, MAX_POINTS], [y_points, y_points], color=line_color, lw=line_width)
    tick_shift, label_shift = get_tick_offsets(points_tick_direction)
    step = max(10, int(MAX_POINTS / 10))
    for p in range(0, MAX_POINTS + step, step):
        ax.plot([p, p], [y_points, y_points + tick_shift], color=line_color, lw=tick_width)
        va = 'top' if label_shift > 0 else 'bottom'
        ax.text(p, y_points + label_shift, str(p), ha='center', va=va,
                fontsize=tick_fontsize, color=text_color)
    ax.text(-left_margin + 5, y_points, "Points", ha='left', va='center',
            fontsize=label_fontsize, color=text_color, fontweight='bold')

    # ---- Total Points 轴 ----
    ax.plot([0, MAX_POINTS], [y_total, y_total], color=line_color, lw=line_width)
    tick_shift, label_shift = get_tick_offsets(others_tick_direction)
    for p in range(0, MAX_POINTS + step, step):
        ax.plot([p, p], [y_total, y_total + tick_shift], color=line_color, lw=tick_width)
        va = 'bottom' if label_shift > 0 else 'top'
        ax.text(p, y_total + label_shift, str(p), ha='center', va=va,
                fontsize=tick_fontsize, color=text_color)
    ax.text(-left_margin + 5, y_total, "Total Points", ha='left', va='center',
            fontsize=label_fontsize, color=text_color, fontweight='bold')

    # ---- Risk 轴（概率） ----
    prob_ticks = np.linspace(0.1, 0.9, 9)
    logit_ticks = np.log(prob_ticks / (1 - prob_ticks))
    point_ticks = (logit_ticks - logit_baseline) * scale
    valid_idx = (point_ticks > 0) & (point_ticks < MAX_POINTS)
    valid_point_ticks = point_ticks[valid_idx]
    valid_prob_ticks = prob_ticks[valid_idx]

    if len(valid_point_ticks) > 0:
        ax.plot([valid_point_ticks.min(), valid_point_ticks.max()],
                [y_prob, y_prob], color=line_color, lw=line_width)
        tick_shift, label_shift = get_tick_offsets(others_tick_direction)
        for pt, p in zip(valid_point_ticks, valid_prob_ticks):
            ax.plot([pt, pt], [y_prob, y_prob + tick_shift], color=line_color, lw=tick_width)
            va = 'bottom' if label_shift > 0 else 'top'
            ax.text(pt, y_prob + label_shift, f"{p:.1f}", ha='center', va=va,
                    fontsize=tick_fontsize, color=text_color)
        ax.text(-left_margin + 5, y_prob, "Risk", ha='left', va='center',
                fontsize=label_fontsize, color=text_color, fontweight='bold')
    else:
        st.warning("概率轴无有效刻度，请检查模型参数。")

    # ---- 在总分轴上标记红点 ----
    # 计算总得分
    total_score = 0.0
    for var in variables:
        raw_val = case[var]
        if var in binary_vars:
            raw_val = 1 if raw_val else 0
        total_score += calc_points(var, raw_val)
    # 截断到有效范围
    total_score_clipped = np.clip(total_score, 0, MAX_POINTS)
    ax.plot(total_score_clipped, y_total, 'ro', markersize=12, markeredgecolor='darkred', zorder=5)
    ax.text(total_score_clipped - 2, y_total + 0.2, f'{total_score:.1f}', fontsize=12,
            color='red', ha='right', va='bottom', fontweight='bold')

    # ---- 在风险轴上标记红点 ----
    prob_case = expit(logit_baseline + total_score / scale)
    # 将概率转换为points位置（用于在Risk轴上画点）
    logit_case = logit_baseline + total_score / scale
    point_prob = (logit_case - logit_baseline) * scale
    point_prob_clipped = np.clip(point_prob, valid_point_ticks.min() if len(valid_point_ticks)>0 else 0,
                                 valid_point_ticks.max() if len(valid_point_ticks)>0 else MAX_POINTS)
    ax.plot(point_prob_clipped, y_prob, 'ro', markersize=12, markeredgecolor='darkred', zorder=5)
    # 显示概率数值
    ax.text(point_prob_clipped + 2, y_prob - 0.1, f'{prob_case:.3f}', fontsize=12,
            color='red', ha='left', va='top', fontweight='bold')

    plt.tight_layout()
    return fig, total_score, prob_case


# ============================================================
# 3. 概率曲线图（与当前工具类似，但使用新的总分体系）
# ============================================================
def plot_probability_curve(total_score, prob_case):
    """绘制总分-概率曲线，并标记当前点"""
    fig, ax = plt.subplots(figsize=(10, 5))
    # 生成曲线
    t_all = np.linspace(0, MAX_POINTS * 1.2, 200)
    logit_all = logit_baseline + t_all / scale
    prob_all = expit(logit_all)
    ax.plot(t_all, prob_all, 'b-', lw=2)
    ax.set_xlabel('Total Points', fontsize=18)
    ax.set_ylabel('Probability', fontsize=18)
    ax.set_xlim(0, MAX_POINTS * 1.2)
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.grid(alpha=0.2)

    # 标记当前点
    ax.plot(total_score, prob_case, 'ro', markersize=12, markeredgecolor='darkred')
    ax.text(total_score + 2, prob_case - 0.02,
            f'Points: {total_score:.1f}\nProb: {prob_case:.3f}',
            fontsize=12, color='red', ha='left', va='top')

    # 可选：标注Youden cutoff（0.351）
    cutoff_prob = 0.351
    cutoff_score = (np.log(cutoff_prob / (1 - cutoff_prob)) - logit_baseline) * scale
    ax.axhline(y=cutoff_prob, color='red', linestyle='--', linewidth=1.5)
    ax.text(-0.4, cutoff_prob-0.01, f'{cutoff_prob:.3f}',
            color='red', fontsize=10, ha='right')
    # ax.axvline(x=cutoff_score, color='red', linestyle='--', linewidth=1.0, alpha=0.5)

    plt.tight_layout()
    return fig

def combine_figures(fig_nomogram, fig_curve, case, total_score, prob, dpi=600):
    try:
        # 1. 将两个图分别转为 PIL Image
        buf1 = BytesIO()
        fig_nomogram.savefig(buf1, format='png', dpi=dpi, bbox_inches='tight')
        buf1.seek(0)
        img1 = Image.open(buf1)

        buf2 = BytesIO()
        fig_curve.savefig(buf2, format='png', dpi=dpi, bbox_inches='tight')
        buf2.seek(0)
        img2 = Image.open(buf2)

        # ========== 可调参数区 ==========
        title_font_size = 50       # 页面主标题
        info_font_size = 36        # 日期、输入、结果
        section_font_size = 36     # 子图标题
        line_height_ratio = 1.3
        spacing_after_title = 10
        spacing_after_date = 10
        spacing_after_input = 10
        spacing_after_result = 20
        spacing_after_line = 10
        spacing_before_img = 10
        spacing_between_sections = 5   # 可以调小到0或5
        bottom_padding = 30
        top_padding = 20
        # =================================

        # 加载字体（优先 Arial，备选 SimHei）
        try:
            title_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", title_font_size)
            info_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", info_font_size)
            section_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", section_font_size)
        except:
            try:
                title_font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", title_font_size)
                info_font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", info_font_size)
                section_font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", section_font_size)
            except:
                # 如果都失败，使用默认字体（但会很小，这里只是保底）
                title_font = info_font = section_font = ImageFont.load_default()

        # 计算行高
        line_height_title = int(title_font_size * line_height_ratio)
        line_height_info = int(info_font_size * line_height_ratio)
        line_height_section = int(section_font_size * line_height_ratio)

        # 顶部信息区高度
        info_height = (top_padding +
                       line_height_title + spacing_after_title +
                       line_height_info + spacing_after_date +
                       line_height_info + spacing_after_input +
                       line_height_info + spacing_after_result +
                       spacing_after_line)

        section_title_height = line_height_section + spacing_before_img
        max_width = max(img1.width, img2.width)
        total_height = (info_height +
                        section_title_height + img1.height +
                        section_title_height + img2.height +
                        spacing_between_sections + bottom_padding)

        combined = Image.new('RGB', (max_width, total_height), 'white')
        draw = ImageDraw.Draw(combined)

        y = top_padding

        # 主标题
        draw.text((20, y), "Extraprostatic Extension Risk Calculator", fill='black', font=title_font)
        y += line_height_title + spacing_after_title

        # 日期
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        draw.text((20, y), f"Date: {now}", fill='black', font=info_font)
        y += line_height_info + spacing_after_date

        # 输入参数
        info = (f"Input: f/tPSA={case['f/tPSA']:.3f}, fPSA={case['fPSA']:.2f} ng/mL, "
                f"CCLmax={case['CCLmax']:.1f} mm, Bulging={int(case['Capsular bulging'])}, "
                f"Disruption={int(case['Capsular disruption'])}, Retraction={int(case['Capsular retraction'])}")
        draw.text((20, y), info, fill='black', font=info_font)
        y += line_height_info + spacing_after_input

        # 预测结果
        result = f"Total Points: {total_score:.1f}, Probability: {prob:.3f}"
        draw.text((20, y), result, fill='black', font=info_font)
        y += line_height_info + spacing_after_result

        # 分隔线
        draw.line((0, y, max_width, y), fill='gray', width=3)
        y += spacing_after_line

        # ---- 列线图（带子标题） ----
        draw.text((20, y), "Nomogram with Current Case", fill='black', font=section_font)
        y += line_height_section + spacing_before_img
        combined.paste(img1, (0, y))
        y += img1.height

        # ---- 概率曲线（带子标题） ----
        y += spacing_between_sections
        draw.text((20, y), "Total Points → Probability Curve", fill='black', font=section_font)
        y += line_height_section + spacing_before_img
        combined.paste(img2, (0, y))

        # 保存为 PNG
        buf_combined = BytesIO()
        combined.save(buf_combined, format='PNG', dpi=(dpi, dpi))
        buf_combined.seek(0)
        return buf_combined

    except Exception as e:
        # 如果出现任何异常，在 Streamlit 中显示错误信息，并返回一张空白图片防止下载按钮失效
        st.error(f"生成图片时出错：{e}")
        # 创建一张空白图片作为 fallback
        dummy = Image.new('RGB', (100, 100), 'white')
        buf = BytesIO()
        dummy.save(buf, format='PNG')
        buf.seek(0)
        return buf
    
# ============================================================
# 4. Streamlit 界面
# ============================================================
st.set_page_config(page_title="Extraprostatic Extension Risk Calculator", layout="wide")
st.title("Extraprostatic Extension Risk Calculator")
st.markdown("Predict the probability of extraprostatic extension using MRI semantic features and clinical variables")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Clinical Variables")
    # ===== 修改点 2：界面输入标签 =====
    ftpsa = st.number_input("Free/Total PSA", min_value=0.01, max_value=1.0, value=0.153, step=0.01, format="%.3f")
    fpsa = st.number_input("Free PSA (ng/mL)", min_value=0.1, max_value=30.0, value=2.56, step=0.1, format="%.2f")
   

with col2:
    st.subheader("MRI Semantic Features")
     # 将 CCLmax 移到临床变量中（原属 MRI 特征，但通常归为临床）
    cclmax = st.number_input("Maximum Capsular Contact Length(CCLmax, mm)", min_value=0.0, max_value=200.0, value=15.4, step=1.0, format="%.1f")
    bulge = st.checkbox("Capsular Bulging", value=False)
    disruption = st.checkbox("Capsular Disruption", value=False)
    retraction = st.checkbox("Capsular Retraction", value=False)

# 构建病例字典
case = {
    "f/tPSA": ftpsa,
    "fPSA": fpsa,
    "CCLmax": cclmax,
    "Capsular bulging": bulge,
    "Capsular disruption": disruption,
    "Capsular retraction": retraction
}

# 绘制列线图并获取结果
fig_nomogram, total_score, prob = plot_nomogram(case)

# 显示指标
st.markdown("---")
col_score, col_prob, col_risk = st.columns(3)
col_score.metric("Total Points", f"{total_score:.1f}")
col_prob.metric("Probability", f"{prob:.3f}")
if prob >= 0.351:
    col_risk.error("High Risk of EPE (≥ 0.351)")
else:
    col_risk.success("Low Risk of EPE (< 0.351)")

# 显示列线图
st.subheader("Nomogram with Current Case Marked")
st.pyplot(fig_nomogram)

# 显示概率曲线
st.subheader("Total Points → Probability Curve")
fig_curve = plot_probability_curve(total_score, prob)
st.pyplot(fig_curve)

# 下载按钮
buf = combine_figures(fig_nomogram, fig_curve, case, total_score, prob, dpi=600)
st.download_button(
    label="📥 Download Full Image (600 DPI)",
    data=buf,
    file_name="nomogram_curve.png",
    mime="image/png"
)

st.caption("* All score calculations follow the nomogram scaling (baseline to maximum risk = 100 points, adjusted for 0.9 probability). The red dots on the nomogram indicate the contribution of each variable and the resulting total points and risk.")
