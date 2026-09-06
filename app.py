import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from datetime import datetime
from io import BytesIO
from PIL import Image

# ============================================================
# 全局字体设置（Arial）和模型参数（完全不变）
# ============================================================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# ----- 模型参数（固定，不可修改）-----
intercept = -1.859
coef = {
    "f/tPSA": -0.368,
    "fPSA": 0.377,
    "mCCL": 1.054,                     # 原 CCLmax
    "Capsular Bulge": 1.095,           # 原 Capsular bulging（首字母大写）
    "Capsular loss": 1.107,
    "Capsular retraction": 0.701
}
mu_fpsa = 0.6659
sigma_fpsa = 0.9259
mu_ftpsa = -2.0339
sigma_ftpsa = 0.7558
mu_cc = 34.566
sigma_cc = 35.530
ranges = {
    "f/tPSA": (0.04, 0.77),
    "fPSA": (0.49, 21.88),
    "mCCL": (0.0, 105.0),              # 原 CCLmax
    "Capsular Bulge": (0, 1),          # 原 Capsular bulging
    "Capsular loss": (0, 1),
    "Capsular retraction": (0, 1)
}
binary_vars = ["Capsular Bulge", "Capsular loss", "Capsular retraction"]  # 原 Capsular bulging
var_display_names = {
    "f/tPSA": "Free/Total PSA",
    "fPSA": "Free PSA",
    "mCCL": "mCCL",                    # 原 CCLmax
    "Capsular Bulge": "Capsular Bulge", # 原 Capsular Bulging
    "Capsular loss": "Capsular Loss",
    "Capsular retraction": "Capsular Retraction"
}

# ----- 分数计算函数（不变，仅修改判断变量名）-----
def get_lp(var, x_raw):
    if var == "mCCL":                  # 原 CCLmax
        return coef[var] * (x_raw - mu_cc) / sigma_cc
    elif var == "fPSA":
        x_log = np.log(x_raw) if x_raw > 0 else np.log(1e-6)
        return coef[var] * (x_log - mu_fpsa) / sigma_fpsa
    elif var == "f/tPSA":
        x_log = np.log(x_raw) if x_raw > 0 else np.log(1e-6)
        return coef[var] * (x_log - mu_ftpsa) / sigma_ftpsa
    else:
        return coef[var] * x_raw

baseline_values = {}
for var in coef:
    if var == "f/tPSA":
        baseline_values[var] = ranges[var][1]
    else:
        baseline_values[var] = ranges[var][0]

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
MAX_POINTS = int(np.ceil(MAX_POINTS / 10) * 10)

def calc_points(var, x_raw):
    base_lp = get_lp(var, baseline_values[var])
    return (get_lp(var, x_raw) - base_lp) * scale

def inv_calc_points(var, points):
    base_lp = get_lp(var, baseline_values[var])
    lp_val = points / scale + base_lp
    if var == "mCCL":                  # 原 CCLmax
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
# 绘图函数（列线图和概率曲线，保持比例不变）
# ============================================================
def plot_nomogram(case):
    # 根据MAX_POINTS动态调整left_margin使轴长度占2/3
    left_margin = int((MAX_POINTS + 10) / 2)  # 使左侧变量名区域约占1/3
    figsize = (12, 8)   # 保持宽高比
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

    label_fontsize = 14
    tick_fontsize = 12
    title_fontsize = 18
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
        "mCCL": y_cc,                     # 原 CCLmax
        "Capsular Bulge": y_bulge,        # 原 Capsular bulging
        "Capsular loss": y_disrupt,
        "Capsular retraction": y_retract
    }

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-left_margin, MAX_POINTS + 10)
    ax.set_ylim(-0.5, y_points + 1.5)
    ax.axis('off')

    # 图形标题（左对齐）
    ax.text(-left_margin + 5, y_points + 0.7, "Nomogram with Current Case Marked",
            fontsize=title_fontsize, fontweight='bold', ha='left', va='bottom',
            color='black', family='Arial')

    def get_tick_offsets(direction):
        if direction == 'up':
            return tick_length, -label_offset
        else:
            return -tick_length, label_offset

    for var in variables:
        y = var_y[var]
        xmin, xmax = ranges[var]
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

        ax.plot([tick_points.min(), tick_points.max()], [y, y],
                color=line_color, lw=line_width)
        tick_shift, label_shift = get_tick_offsets(others_tick_direction)
        for t, tp in zip(ticks, tick_points):
            ax.plot([tp, tp], [y, y + tick_shift], color=line_color, lw=tick_width)
            if var in binary_vars:
                label = f"{int(t)}"
            else:
                label = f"{t:.1f}" if not np.isclose(t, 0.0, atol=1e-6) else "0"
            va = 'bottom' if label_shift > 0 else 'top'
            ax.text(tp, y + label_shift, label, ha='center', va=va,
                    fontsize=tick_fontsize, color=text_color)

        display_name = var_display_names.get(var, var)
        ax.text(-left_margin + 5, y, display_name, ha='left', va='center',
                fontsize=label_fontsize, color=text_color, fontweight='bold')

        raw_val = case[var]
        if var in binary_vars:
            raw_val = 1 if raw_val else 0
        score = calc_points(var, raw_val)
        ax_range_min = tick_points.min()
        ax_range_max = tick_points.max()
        x_red = np.clip(score, ax_range_min, ax_range_max)
        ax.plot(x_red, y, 'ro', markersize=8, markeredgecolor='darkred', zorder=5)
        ax.text(x_red - 2, y + 0.2, f'{score:.1f}', fontsize=10, color='red',
                ha='right', va='bottom', fontweight='bold')

    # Points轴
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

    # Total Points轴
    ax.plot([0, MAX_POINTS], [y_total, y_total], color=line_color, lw=line_width)
    tick_shift, label_shift = get_tick_offsets(others_tick_direction)
    for p in range(0, MAX_POINTS + step, step):
        ax.plot([p, p], [y_total, y_total + tick_shift], color=line_color, lw=tick_width)
        va = 'bottom' if label_shift > 0 else 'top'
        ax.text(p, y_total + label_shift, str(p), ha='center', va=va,
                fontsize=tick_fontsize, color=text_color)
    ax.text(-left_margin + 5, y_total, "Total Points", ha='left', va='center',
            fontsize=label_fontsize, color=text_color, fontweight='bold')

    # Risk轴
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

    total_score = 0.0
    for var in variables:
        raw_val = case[var]
        if var in binary_vars:
            raw_val = 1 if raw_val else 0
        total_score += calc_points(var, raw_val)
    total_score_clipped = np.clip(total_score, 0, MAX_POINTS)
    ax.plot(total_score_clipped, y_total, 'ro', markersize=10, markeredgecolor='darkred', zorder=5)
    ax.text(total_score_clipped - 2, y_total + 0.2, f'{total_score:.1f}', fontsize=10,
            color='red', ha='right', va='bottom', fontweight='bold')

    prob_case = expit(logit_baseline + total_score / scale)
    logit_case = logit_baseline + total_score / scale
    point_prob = (logit_case - logit_baseline) * scale
    point_prob_clipped = np.clip(point_prob, valid_point_ticks.min() if len(valid_point_ticks)>0 else 0,
                                 valid_point_ticks.max() if len(valid_point_ticks)>0 else MAX_POINTS)
    ax.plot(point_prob_clipped, y_prob, 'ro', markersize=10, markeredgecolor='darkred', zorder=5)
    ax.text(point_prob_clipped + 2, y_prob - 0.1, f'{prob_case:.3f}', fontsize=10,
            color='red', ha='left', va='top', fontweight='bold')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    return fig, total_score, prob_case

def plot_probability_curve(total_score, prob_case):
    fig, ax = plt.subplots(figsize=(10, 8))   # 与列线图尺寸一致
    t_all = np.linspace(0, MAX_POINTS * 1.2, 200)
    logit_all = logit_baseline + t_all / scale
    prob_all = expit(logit_all)
    ax.plot(t_all, prob_all, 'b-', lw=2)
    ax.set_xlabel('Total Points', fontsize=14)
    ax.set_ylabel('Probability', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_xlim(0, MAX_POINTS * 1.2)
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.grid(alpha=0.2)

    ax.set_title("Total Points → Probability Curve", fontsize=18, fontweight='bold',
                 pad=15, family='Arial', loc='left')

    ax.plot(total_score, prob_case, 'ro', markersize=10, markeredgecolor='darkred')
    ax.text(total_score + 2, prob_case - 0.02,
            f'Points: {total_score:.1f}\nProb: {prob_case:.3f}',
            fontsize=10, color='red', ha='left', va='top')

    cutoff_prob = 0.351
    cutoff_score = (np.log(cutoff_prob / (1 - cutoff_prob)) - logit_baseline) * scale
    ax.axhline(y=cutoff_prob, color='red', linestyle='--', linewidth=1.5)
    ax.text(-0.4, cutoff_prob-0.01, f'{cutoff_prob:.3f}',
            color='red', fontsize=9, ha='right')

    plt.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.12)
    return fig

# ============================================================
# 下载图片生成（保持不变，仅修改信息字符串中的变量名）
# ============================================================
def combine_figures(fig_nomogram, fig_curve, case, total_score, prob, dpi=600):
    try:
        buf1 = BytesIO()
        fig_nomogram.savefig(buf1, format='png', dpi=dpi, bbox_inches='tight')
        buf1.seek(0)
        img1 = Image.open(buf1)
        buf2 = BytesIO()
        fig_curve.savefig(buf2, format='png', dpi=dpi, bbox_inches='tight')
        buf2.seek(0)
        img2 = Image.open(buf2)

        nomogram_width_inch = fig_nomogram.get_size_inches()[0]
        lines = [
            ("Extraprostatic Extension Risk Calculator", 30, 'bold'),
            (f"Date: {datetime.now().strftime('%Y-%m-%d')}", 20, 'normal'),
            (f"Clinical: f/tPSA={case['f/tPSA']:.3f}, fPSA={case['fPSA']:.2f} ng/mL", 20, 'normal'),
            (f"MRI: mCCL={case['mCCL']:.1f} mm, Bulge={int(case['Capsular Bulge'])}, "   # 修改此处
             f"Loss={int(case['Capsular loss'])}, Retraction={int(case['Capsular retraction'])}", 20, 'normal'),
            (f"Total Points: {total_score:.1f}, Probability: {prob:.3f}", 20, 'normal')
        ]
        top_margin_extra = 0.08
        line_heights = [30*1.5, 20*1.5, 20*1.5, 20*1.5, 20*1.5]
        total_info_height = sum(line_heights) + 100
        fig_info, ax_info = plt.subplots(figsize=(nomogram_width_inch, total_info_height/100))
        ax_info.axis('off')
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        y_start = 0.95 - top_margin_extra
        y_step = 0.18
        for i, (text, size, weight) in enumerate(lines):
            ax_info.text(0.01, y_start - i*y_step, text, fontsize=size, va='top',
                        fontweight=weight, family='Arial')
        ax_info.axhline(y=0.05, color='gray', linewidth=2)
        plt.tight_layout(pad=0.1)
        buf_info = BytesIO()
        fig_info.savefig(buf_info, format='png', dpi=dpi, bbox_inches='tight')
        buf_info.seek(0)
        img_info = Image.open(buf_info)
        plt.close(fig_info)

        max_width = max(img_info.width, img1.width, img2.width)
        total_height = img_info.height + img1.height + img2.height + 20
        combined = Image.new('RGB', (max_width, total_height), 'white')
        y_offset = 0
        combined.paste(img_info, (0, y_offset))
        y_offset += img_info.height + 10
        combined.paste(img1, (0, y_offset))
        y_offset += img1.height + 10
        combined.paste(img2, (0, y_offset))
        buf_combined = BytesIO()
        combined.save(buf_combined, format='PNG', dpi=(dpi, dpi))
        buf_combined.seek(0)
        return buf_combined
    except Exception as e:
        st.error(f"生成图片时出错：{e}")
        import traceback
        st.error(traceback.format_exc())
        dummy = Image.new('RGB', (100, 100), 'white')
        buf = BytesIO()
        dummy.save(buf, format='PNG')
        buf.seek(0)
        return buf

# ============================================================
# Streamlit 界面（严格按照用户指定布局）
# ============================================================
st.set_page_config(page_title="Extraprostatic Extension Risk Calculator", layout="wide")

# ---------- 自定义 CSS 实现边距、对齐和字体 ----------
st.markdown("""
<style>
    /* 全局边距 2.5cm */
    .main-container {
        padding: 2.5cm !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }
    /* 标题样式 */
    .main-title {
        font-family: Arial, sans-serif;
        font-size: 60px !important;
        font-weight: bold !important;
        text-align: center !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .date-right {
        font-family: Arial, sans-serif;
        font-size: 20px !important;
        text-align: right !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .subtitle-left {
        font-family: Arial, sans-serif;
        font-size: 20px !important;
        text-align: left !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .section-title {
        font-family: Arial, sans-serif;
        font-size: 40px !important;
        font-weight: bold !important;
        margin: 0.2rem 0 !important;
    }
    .label-text {
        font-family: Arial, sans-serif;
        font-size: 30px !important;
        font-weight: bold !important;
    }
    /* 输入框内部文字大小 */
    .stNumberInput input, .stTextInput input {
        height: 40px !important;
        font-size: 20px !important;
    }
    /* 统一所有标签字体（数字输入框和 checkbox） */
    .stNumberInput label,
    .stCheckbox label {
        font-size: 30px !important;
        font-weight: bold !important;
    }
    /* 下移 checkbox 使其与输入框对齐 */
    .stCheckbox {
        margin-top: 32px !important;
    }
    /* 分割线 */
    hr {
        margin: 1rem 0 !important;
    }
    /* 风险框 */
    .risk-box {
        height: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    /* 指标卡样式 */
    .stMetric {
        font-size: 30px !important;
        font-weight: bold !important;
    }
    .stMetric label {
        font-size: 20px !important;
        font-weight: bold !important;
    }
    /* 列线图和概率曲线图容器 */
    .figure-container {
        width: 100%;
        height: auto;
    }
</style>
""", unsafe_allow_html=True)

# ---------- 页面内容 ----------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# 第一行：标题（居中，30号，加粗）
st.markdown('<p class="main-title">Extraprostatic Extension Risk Calculator</p>', unsafe_allow_html=True)

# 第二行：日期（右对齐，10号）
current_date = datetime.now().strftime("%Y-%m-%d")
st.markdown(f'<p class="date-right">{current_date}</p>', unsafe_allow_html=True)

# 第三行：副标题（左对齐，10号）
st.markdown('<p class="subtitle-left">Predict the probability of extraprostatic extension using MRI semantic features and clinical variables</p>', unsafe_allow_html=True)

# 第四行：两列（Clinical Variables 和 MRI Semantic Features）
col1, col2 = st.columns(2)
with col1:
    st.markdown('<p class="section-title">Clinical Variables</p>', unsafe_allow_html=True)
    ftpsa = st.number_input("Free/Total PSA", min_value=0.01, max_value=1.0, value=0.153, step=0.01, format="%.3f", key="ftpsa")
    fpsa = st.number_input("Free PSA (ng/mL)", min_value=0.1, max_value=30.0, value=2.56, step=0.1, format="%.3f", key="fpsa")
with col2:
    st.markdown('<p class="section-title">MRI Semantic Features</p>', unsafe_allow_html=True)
    # 修改输入框标签为 Multiplanar Capsular Contact Length (mCCL, mm)
    mCCL = st.number_input("Multiplanar Capsular Contact Length (mCCL, mm)", min_value=0.0, max_value=200.0, value=15.4, step=1.0, format="%.1f", key="mCCL")
    # 三个checkbox横向排列，修改标签 Capsular Bulging -> Capsular Bulge
    col_b, col_d, col_r = st.columns(3)
    with col_b:
        bulge = st.checkbox("Capsular Bulge", value=False)   # 原 Capsular Bulging
    with col_d:
        disruption = st.checkbox("Capsular Loss", value=False)
    with col_r:
        retraction = st.checkbox("Capsular Retraction", value=False)

# 构建病例字典（键名已更新）
case = {
    "f/tPSA": ftpsa,
    "fPSA": fpsa,
    "mCCL": mCCL,                     # 原 CCLmax
    "Capsular Bulge": bulge,          # 原 Capsular bulging
    "Capsular loss": disruption,
    "Capsular retraction": retraction
}

# 计算并绘制
fig_nomogram, total_score, prob = plot_nomogram(case)
fig_curve = plot_probability_curve(total_score, prob)

# ---------- 分割线 ----------
st.markdown("---")

# ---------- 总分、概率和风险框（三列等宽） ----------
col_score, col_prob, col_risk = st.columns(3)
with col_score:
    st.metric(label="Total Points", value=f"{total_score:.1f}")
with col_prob:
    st.metric(label="Probability", value=f"{prob:.3f}")
with col_risk:
    if prob >= 0.351:
        st.error("High Risk of EPE (≥ 0.351)")
    else:
        st.success("Low Risk of EPE (< 0.351)")

# ---------- 列线图（占2/3轴长，1/3图例已在图形内实现） ----------
st.pyplot(fig_nomogram, use_container_width=True)

# ---------- 概率曲线 ----------
st.pyplot(fig_curve, use_container_width=True)

# ---------- 下载按钮 ----------
buf = combine_figures(fig_nomogram, fig_curve, case, total_score, prob, dpi=600)
st.download_button(
    label="📥 Download Full Image (600 DPI)",
    data=buf,
    file_name=f"nomogram_curve_{datetime.now().strftime('%Y%m%d')}.png",
    mime="image/png"
)

st.caption("* All score calculations follow the nomogram scaling (baseline to maximum risk = 100 points, adjusted for 0.9 probability). The red dots on the nomogram indicate the contribution of each variable and the resulting total points and risk.")

st.markdown('</div>', unsafe_allow_html=True)
