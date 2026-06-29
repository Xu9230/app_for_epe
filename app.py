import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

# ========================
# 模型参数（与你的原始代码完全一致）
# ========================
intercept = -1.859
coef = {
    "f/tPSA": -0.368,
    "fPSA": 0.377,
    "CCLmax": 1.054,
    "Capsular bulging": 1.095,
    "Capsular disruption": 1.107,
    "Capsular retraction": 0.701
}

mu_fpsa = 0.6659
sigma_fpsa = 0.9259
mu_ftpsa = -2.0339
sigma_ftpsa = 0.7558
mu_cc = 34.566
sigma_cc = 35.530

base_coef = abs(coef["f/tPSA"])   # 0.368
base_score = 10
score_unit = base_coef / base_score   # 0.0368

binary_scores = {
    "Capsular bulging":    coef["Capsular bulging"]    / score_unit,
    "Capsular disruption": coef["Capsular disruption"] / score_unit,
    "Capsular retraction": coef["Capsular retraction"] / score_unit
}

fpsa_per_unit = coef["fPSA"] / sigma_fpsa / score_unit
ftpsa_per_unit = coef["f/tPSA"] / sigma_ftpsa / score_unit
cc_per_mm = coef["CCLmax"] / sigma_cc / score_unit

fpsa_range = (0.49, 21.88)
ftpsa_range = (0.04, 0.77)
cc_range = (0.0, 105.0)

baseline = {
    "f/tPSA": ftpsa_range[1],
    "fPSA": fpsa_range[0],
    "CCLmax": cc_range[0],
    "Capsular bulging": 0,
    "Capsular disruption": 0,
    "Capsular retraction": 0
}

def safe_log(x):
    return np.log(x) if x > 0 else np.log(1e-6)

def lp_contribution(var, x):
    if var == "CCLmax":
        return coef[var] * (x - mu_cc) / sigma_cc
    elif var == "fPSA":
        return coef[var] * (safe_log(x) - mu_fpsa) / sigma_fpsa
    elif var == "f/tPSA":
        return coef[var] * (safe_log(x) - mu_ftpsa) / sigma_ftpsa
    else:
        return coef[var] * x

lp_baseline = sum(lp_contribution(v, baseline[v]) for v in coef)
logit_baseline = lp_baseline + intercept

worst = {
    "f/tPSA": ftpsa_range[0],
    "fPSA": fpsa_range[1],
    "CCLmax": cc_range[1],
    "Capsular bulging": 1,
    "Capsular disruption": 1,
    "Capsular retraction": 1
}
lp_worst = sum(lp_contribution(v, worst[v]) for v in coef)
max_score = (lp_worst - lp_baseline) / score_unit

def calc_score(var, x):
    base = lp_contribution(var, baseline[var])
    return (lp_contribution(var, x) - base) / score_unit

def plot_score_chart(case):
    """根据输入的病例绘制评分系统图，返回 matplotlib figure"""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 18
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 7),
                                            gridspec_kw={'height_ratios': [1.5, 1]})

    # ---- 上部：变量轴 ----
    ax_top.set_xlim(-15, 115)
    ax_top.set_ylim(0, 6.5)
    ax_top.axis('off')
    ax_top.set_title('Variable → Score Assignment', fontsize=14, fontweight='bold', pad=12)

    y_pos = [5.5, 4.5, 3.5, 2.5, 1.5, 0.5]
    var_names = ["f/tPSA", "fPSA", "CCLmax", "Capsular bulging", "Capsular disruption", "Capsular retraction"]
    var_labels = ["f/tPSA", "fPSA", "CCLmax", "Capsular bulging", "Capsular disruption", "Capsular retraction"]

    axis_score_ranges = {}

    # 二分类变量轴
    for i, var in enumerate(var_names[3:], start=3):
        y = y_pos[i]
        pts_present = binary_scores[var]
        pts_absent = 0.0
        axis_score_ranges[var] = (pts_absent, pts_present, 0, 100)
        pts_int = round(pts_present)
        ax_top.plot([0, 100], [y, y], 'k-', lw=1.5)
        ax_top.scatter(0, y, s=60, color='black', zorder=3)
        ax_top.text(0, y - 0.3, 'Absent', ha='center', va='top', fontsize=10, color='gray')
        ax_top.scatter(100, y, s=60, color='black', zorder=3)
        ax_top.text(100, y - 0.3, 'Present', ha='center', va='top', fontsize=10, color='gray')
        ax_top.text(-10, y, var_labels[i], ha='right', va='center', fontsize=14)
        ax_top.text(103, y, f'{pts_int} pts', ha='left', va='center', fontsize=14, fontweight='bold')

    # CCLmax 轴
    cc_y = y_pos[2]
    cc_ticks = np.linspace(cc_range[0], cc_range[1], 5)
    cc_scores = [calc_score("CCLmax", v) for v in cc_ticks]
    score_min_cc, score_max_cc = min(cc_scores), max(cc_scores)
    axis_score_ranges["CCLmax"] = (score_min_cc, score_max_cc, 0, 100)
    cc_x = [(v - cc_range[0]) / (cc_range[1] - cc_range[0]) * 100 for v in cc_ticks]
    ax_top.plot([0, 100], [cc_y, cc_y], 'k-', lw=1.5)
    for xp, val, sc in zip(cc_x, cc_ticks, cc_scores):
        ax_top.scatter(xp, cc_y, s=60, color='black', zorder=3)
        label = '0' if val == 0 else f'{val:.0f}'
        ax_top.text(xp, cc_y - 0.3, label, ha='center', va='top', fontsize=10, color='gray')
        if val == cc_range[1]:
            ax_top.text(103, cc_y, f'{sc:.1f} pts', ha='left', va='center', fontsize=14, fontweight='bold')
    ax_top.text(-10, cc_y, 'CCLmax', ha='right', va='center', fontsize=14)

    # fPSA 轴
    fpsa_y = y_pos[1]
    fpsa_ticks = np.linspace(fpsa_range[0], fpsa_range[1], 4)
    fpsa_scores = [calc_score("fPSA", v) for v in fpsa_ticks]
    score_min_fpsa, score_max_fpsa = min(fpsa_scores), max(fpsa_scores)
    axis_score_ranges["fPSA"] = (score_min_fpsa, score_max_fpsa, 0, 100)
    fpsa_x = [(v - fpsa_range[0]) / (fpsa_range[1] - fpsa_range[0]) * 100 for v in fpsa_ticks]
    ax_top.plot([0, 100], [fpsa_y, fpsa_y], 'k-', lw=1.5)
    for xp, val, sc in zip(fpsa_x, fpsa_ticks, fpsa_scores):
        ax_top.scatter(xp, fpsa_y, s=60, color='black', zorder=3)
        ax_top.text(xp, fpsa_y - 0.3, f'{val:.1f}', ha='center', va='top', fontsize=10, color='gray')
        if val == fpsa_range[1]:
            ax_top.text(103, fpsa_y, f'{sc:.1f} pts', ha='left', va='center', fontsize=14, fontweight='bold')
    ax_top.text(-10, fpsa_y, 'fPSA', ha='right', va='center', fontsize=14)

    # f/tPSA 轴（反转）
    ftpsa_y = y_pos[0]
    ftpsa_ticks = np.linspace(ftpsa_range[1], ftpsa_range[0], 4)
    ftpsa_scores = [calc_score("f/tPSA", v) for v in ftpsa_ticks]
    score_min_ftpsa, score_max_ftpsa = min(ftpsa_scores), max(ftpsa_scores)
    axis_score_ranges["f/tPSA"] = (score_min_ftpsa, score_max_ftpsa, 0, 100)
    ftpsa_x = [(v - ftpsa_range[1]) / (ftpsa_range[0] - ftpsa_range[1]) * 100 for v in ftpsa_ticks]
    ax_top.plot([0, 100], [ftpsa_y, ftpsa_y], 'k-', lw=1.5)
    for xp, val, sc in zip(ftpsa_x, ftpsa_ticks, ftpsa_scores):
        ax_top.scatter(xp, ftpsa_y, s=60, color='black', zorder=3)
        ax_top.text(xp, ftpsa_y - 0.3, f'{val:.2f}', ha='center', va='top', fontsize=10, color='gray')
        if val == ftpsa_ticks[0]:
            ax_top.text(xp, ftpsa_y + 0.4, f'{sc:.1f} pts', ha='center', va='bottom', fontsize=11, color='gray')
        if val == ftpsa_ticks[-1]:
            ax_top.text(103, ftpsa_y, f'{sc:.1f} pts', ha='left', va='center', fontsize=14, fontweight='bold')
    ax_top.text(-10, ftpsa_y, 'f/tPSA', ha='right', va='center', fontsize=14)

    # ----- 标出病例红点及得分 -----
    total_score = 0.0
    for var in var_names:
        y = y_pos[var_names.index(var)]
        raw_val = case[var]
        if var in binary_scores:
            raw_val = 1 if raw_val else 0
        score = calc_score(var, raw_val)
        total_score += score
        smin, smax, xmin, xmax = axis_score_ranges[var]
        x_red = xmin + (np.clip(score, smin, smax) - smin) / (smax - smin) * (xmax - xmin)
        ax_top.plot(x_red, y, 'ro', markersize=10, markeredgecolor='darkred', zorder=5)
        ax_top.text(x_red - 0.75, y + 0.15, f'{score:.1f}', fontsize=12, color='red', ha='right', va='bottom')

    # ----- 下部：总分–概率曲线 -----
    ax_bottom.set_title('Total Score → Probability', fontsize=18, fontweight='bold')
    t_all = np.linspace(0, max_score + 10, 200)
    logit_all = logit_baseline + t_all * score_unit
    prob_all = expit(logit_all)
    ax_bottom.plot(t_all, prob_all, 'b-', lw=2)
    ax_bottom.set_xlabel('Total Score', fontsize=18)
    ax_bottom.set_ylabel('Probability', fontsize=18)
    ax_bottom.set_xlim(0, 250)
    ax_bottom.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_bottom.set_ylim(0, 1.02)
    ax_bottom.grid(alpha=0.2)

    cutoff_prob = 0.351
    cutoff_score = (np.log(cutoff_prob / (1 - cutoff_prob)) - logit_baseline) / score_unit
    ax_bottom.axhline(y=cutoff_prob, color='red', linestyle='--', linewidth=1.5)
    ax_bottom.text(-15, cutoff_prob, f'{cutoff_prob:.3f}', color='red', fontsize=12, va='center')
    ax_bottom.text(245, -0.35, f'Cutoff {cutoff_prob:.3f} is selected with the Youden J index',
                   fontsize=9, ha='right', va='top', color='gray')

    for p in np.arange(0.1, 1.01, 0.1):
        t_p = (np.log(p / (1 - p)) - logit_baseline) / score_unit
        if 0 < t_p <= max_score:
            ax_bottom.plot(t_p, p, 'ko', markersize=8)
            ax_bottom.text(t_p - 2.5, p - 0.05, f'{t_p:.0f} pts', fontsize=10, ha='left', va='top')

    prob_case = expit(logit_baseline + total_score * score_unit)
    ax_bottom.plot(total_score, prob_case, 'ro', markersize=12, markerfacecolor='red', markeredgecolor='darkred')
    ax_bottom.text(total_score - 1.5, prob_case + 0.03,
                   f'{total_score:.1f} pts, {prob_case:.3f}',
                   fontsize=12, color='red', fontweight='bold', ha='right', va='bottom')

    plt.tight_layout()
    return fig, total_score, prob_case

# ========================
# Streamlit 界面
# ========================
st.set_page_config(page_title="Extraprostatic Extension Risk Calculator", layout="wide")
st.title("Extraprostatic Extension Risk Calculator")
st.markdown("Predict the probability of extraprostatic extension using MRI semantic features and clinical variables")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Clinical Variables")
    ftpsa = st.number_input("free/total PSA", min_value=0.01, max_value=1.0, value=0.153, step=0.01, format="%.3f")
    fpsa = st.number_input("free PSA (ng/mL)", min_value=0.1, max_value=30.0, value=2.56, step=0.1, format="%.2f")
    cclmax = st.number_input("capsular contact length (CCLmax, mm)", min_value=0.0, max_value=200.0, value=15.4, step=1.0, format="%.1f")

with col2:
    st.subheader("MRI Semantic Features")
    bulge = st.checkbox("Capsular bulging", value=False)
    disruption = st.checkbox("Capsular disruption", value=False)
    retraction = st.checkbox("Capsular retraction", value=False)

# 构建病例字典
case = {
    "f/tPSA": ftpsa,
    "fPSA": fpsa,
    "CCLmax": cclmax,
    "Capsular bulging": bulge,
    "Capsular disruption": disruption,
    "Capsular retraction": retraction
}

# 绘图并获取结果
fig, total_score, prob = plot_score_chart(case)

st.markdown("---")
col_score, col_prob, col_risk = st.columns(3)
col_score.metric("Total Score", f"{total_score:.1f}")
col_prob.metric("Probability", f"{prob:.3f}")
if prob >= 0.351:
    col_risk.error("High Risk of EPE (≥ 0.351)")
else:
    col_risk.success("Low Risk of EPE (< 0.351)")

st.pyplot(fig)
st.markdown("* CCLmax range (0-105) corresponds to the 2.5%-97.5% percentile of the training set.\Binary variable scores are rounded integers; actual calculations use exact regression coefficients.")
