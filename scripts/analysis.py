from scipy.stats import wilcoxon
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Wilcoxon test
# ------------------------------------------------------------

def compute_wilcoxon(
    csv_path,
    pref_baseline="pref_internal",
    pref_steered="pref_steered"
):
    df = pd.read_csv(csv_path)

    # lexical item
    df["lex_item"] = df["weak"] + "–" + df["strong"]

    # pragmatic win-rate
    base_wr = (
        df.groupby("lex_item")[pref_baseline]
          .mean()
          .rename("baseline")
    )

    st_wr = (
        df.groupby("lex_item")[pref_steered]
          .mean()
          .rename("steered")
    )

    # align
    df_merge = pd.concat([base_wr, st_wr], axis=1).dropna()

    # difference (steered - baseline)
    diff = df_merge["steered"] - df_merge["baseline"]

    # Wilcoxon signed-rank test
    stat, p = wilcoxon(diff)

    return stat, p, df_merge


### run ###
stat_fixed, p_fixed_w, df_fixed_w = compute_wilcoxon(
    csv_path="results_uniform.csv",
    pref_baseline="pref_internal",
    pref_steered="pref_steered"
)

print("=== Wilcoxon (Baseline vs Uniform α) ===")
print(f"stat = {stat_fixed:.3f}, p = {p_fixed_w:.4g}")


stat_grad, p_grad_w, df_grad_w = compute_wilcoxon(
    csv_path="results_grade.csv",
    pref_baseline="pref_internal",
    pref_steered="pref_steered"
)

print("=== Wilcoxon (Baseline vs Graded α) ===")
print(f"stat = {stat_grad:.3f}, p = {p_grad_w:.4g}")




# ------------------------------------------------------------
# Spearson test
# ------------------------------------------------------------

def compute_spearman(
    baseline_csv,
    steered_csv,
    pref_baseline="pref_internal",
    pref_steered="pref_steered"
):

    df_base = pd.read_csv(baseline_csv)
    df_st   = pd.read_csv(steered_csv)

    # lexical item
    df_base["lex_item"] = df_base["weak"] + "–" + df_base["strong"]
    df_st["lex_item"]   = df_st["weak"] + "–" + df_st["strong"]

    # pragmatic win-rate
    base_wr = (
        df_base.groupby("lex_item")[pref_baseline]
               .mean()
               .rename("baseline")
    )
    st_wr = (
        df_st.groupby("lex_item")[pref_steered]
             .mean()
             .rename("steered")
    )

    # align
    df_merge = pd.concat([base_wr, st_wr], axis=1).dropna()

    rho, p = spearmanr(df_merge["baseline"], df_merge["steered"])

    return rho, p, df_merge


### run ###
rho_fixed, p_fixed, df_fixed = compute_spearman(
    baseline_csv="results_uniform.csv",
    steered_csv="results_uniform.csv",
    pref_baseline="pref_internal",
    pref_steered="pref_steered"
)

print("=== Spearman (Baseline vs Uniform α) ===")
print(f"ρ = {rho_fixed:.3f}, p = {p_fixed:.4g}")


rho_grad, p_grad, df_grad = compute_spearman(
    baseline_csv="results_grade.csv",
    steered_csv="results_grade.csv",
    pref_baseline="pref_internal",
    pref_steered="pref_steered"
)

print("=== Spearman (Baseline vs Graded α) ===")
print(f"ρ = {rho_grad:.3f}, p = {p_grad:.4g}")




# ------------------------------------------------------------
# Grade-wise mean/SD change
# ------------------------------------------------------------

def grade_mean_abs_diff(csv_path):
    df = pd.read_csv(csv_path)

    results = {}

    for g in ["A", "B", "C", "D", "E"]:
        sub = df[df["grade"] == g]

        if len(sub) == 0:
            continue

        diffs = np.abs(sub["pref_steered"] - sub["pref_internal"])

        results[g] = {
            "mean_abs_diff": diffs.mean(),
            "std_abs_diff": diffs.std(),
            "n": len(diffs)
        }

    return results



### run ###
print("=== Grade-wise mean |Δ| (Uniform) ===")
print(grade_mean_abs_diff("results_uniform.csv"))

print("\n=== Grade-wise mean |Δ| (Graded) ===")
print(grade_mean_abs_diff("results_grade.csv"))




# ------------------------------------------------------------
# Visualization (Bar)
# ------------------------------------------------------------

def plot_pragmatic_winrate(csv_path, pref_col, title, bar_color="#1f77b4"):
    df = pd.read_csv(csv_path)
    df["lex_item"] = df["weak"] + "–" + df["strong"]

    win_rate = (
        df.groupby("lex_item")[pref_col]
          .mean()
          .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 20))
    sns.barplot(
        x=win_rate.values,
        y=win_rate.index,
        color=bar_color
    )
    plt.title(title, fontsize=14)
    plt.xlabel("")
    plt.ylabel("")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()



### run ###
plot_pragmatic_winrate(
    "results_uniform.csv",
    pref_col="pref_internal",
    title="Baseline"
)

plot_pragmatic_winrate(
    "results_uniform.csv",
    pref_col="pref_steered",
    title="Uniform Activation Steering"
)

plot_pragmatic_winrate(
    "results_grade.csv",
    pref_col="pref_steered",
    title="Graded Activation Steering"
)




# ------------------------------------------------------------
# Visualization (Plot - Uniform)
# ------------------------------------------------------------

df = pd.read_csv("results_uniform.csv")

plt.style.use("seaborn-v0_8")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ============================
# (1) Baseline Internal similarity
# ============================

ax1 = axes[0]

sns.kdeplot(
    x=df["sim_internal_pragmatic"],
    y=df["sim_internal_logical"],
    fill=True, cmap="Blues", thresh=0.05, levels=10, ax=ax1
)
sns.scatterplot(
    x=df["sim_internal_pragmatic"],
    y=df["sim_internal_logical"],
    s=25, alpha=0.5, ax=ax1
)

lim_min = min(df["sim_internal_pragmatic"].min(), df["sim_internal_logical"].min())
lim_max = max(df["sim_internal_pragmatic"].max(), df["sim_internal_logical"].max())

ax1.plot([lim_min, lim_max], [lim_min, lim_max], "r--")
ax1.set_title("Baseline")
ax1.set_xlabel("")
ax1.set_ylabel("")
ax1.set_xlim(lim_min, lim_max)
ax1.set_ylim(lim_min, lim_max)


# ============================
# (2) Steered (Uniform-alpha)
# ============================

ax2 = axes[1]

sns.kdeplot(
    x=df["sim_steered_pragmatic"],
    y=df["sim_steered_logical"],
    fill=True, cmap="Blues", thresh=0.05, levels=10, ax=ax2
)
sns.scatterplot(
    x=df["sim_steered_pragmatic"],
    y=df["sim_steered_logical"],
    s=25, alpha=0.5, ax=ax2
)

lim_min2 = min(df["sim_steered_pragmatic"].min(), df["sim_steered_logical"].min())
lim_max2 = max(df["sim_steered_pragmatic"].max(), df["sim_steered_logical"].max())

ax2.plot([lim_min2, lim_max2], [lim_min2, lim_max2], "r--")
ax2.set_title("Uniform Activation Steering")
ax2.set_xlabel("")
ax2.set_ylabel("")
ax2.set_xlim(lim_min2, lim_max2)
ax2.set_ylim(lim_min2, lim_max2)

# Label
fig.suptitle("", fontsize=18)
fig.text(0.5, 0.01, "Pragmatic similarity", ha="center", fontsize=12)
fig.text(0.04, 0.5, "Logical similarity", va="center", rotation="vertical", fontsize=12)

### run ###
plt.tight_layout(rect=[0.05, 0.03, 1, 0.93])
plt.show()




# ------------------------------------------------------------
# Visualization (Plot - Graded)
# ------------------------------------------------------------

grades = ["A", "B", "C", "D", "E"]

df_base = pd.read_csv("results_grade.csv")

plt.style.use("seaborn-v0_8")

fig, axes = plt.subplots(2, len(grades), figsize=(20, 8))


# ============================
# (1) Baseline Internal similarity
# ============================

lim_min_base = min(
    df_base["sim_internal_pragmatic"].min(),
    df_base["sim_internal_logical"].min()
)
lim_max_base = max(
    df_base["sim_internal_pragmatic"].max(),
    df_base["sim_internal_logical"].max()
)


# ============================
# (2) Steered (Gradient-alpha)
# ============================

lim_min_st = min(
    df_base["sim_steered_pragmatic"].min(),
    df_base["sim_steered_logical"].min()
)
lim_max_st = max(
    df_base["sim_steered_pragmatic"].max(),
    df_base["sim_steered_logical"].max()
)

for i, g in enumerate(grades):

    sub = df_base[df_base["grade"] == g]

    # -------------------------------------------------
    # (1) Baseline — Row 0, Column i
    # -------------------------------------------------
    ax1 = axes[0, i]

    if len(sub) > 0:
        sns.kdeplot(
            data=sub,
            x="sim_internal_pragmatic",
            y="sim_internal_logical",
            fill=True, cmap="Blues", thresh=0.05, levels=10, ax=ax1
        )
        sns.scatterplot(
            data=sub,
            x="sim_internal_pragmatic",
            y="sim_internal_logical",
            color="black", alpha=0.5, s=25, ax=ax1
        )

    # baseline limits 적용
    ax1.plot([lim_min_base, lim_max_base], [lim_min_base, lim_max_base], "r--", linewidth=1)
    ax1.set_title(f"Grade {g} — Baseline")
    ax1.set_xlim(lim_min_base, lim_max_base)
    ax1.set_ylim(lim_min_base, lim_max_base)
    ax1.set_xlabel("")
    ax1.set_ylabel("")

    # -------------------------------------------------
    # (2) Steering — Row 1, Column i
    # -------------------------------------------------
    ax2 = axes[1, i]

    if len(sub) > 0:
        sns.kdeplot(
            data=sub,
            x="sim_steered_pragmatic",
            y="sim_steered_logical",
            fill=True, cmap="Blues", thresh=0.05, levels=10, ax=ax2
        )
        sns.scatterplot(
            data=sub,
            x="sim_steered_pragmatic",
            y="sim_steered_logical",
            color="black", alpha=0.5, s=25, ax=ax2
        )

    # steering limits
    ax2.plot([lim_min_st, lim_max_st], [lim_min_st, lim_max_st], "r--", linewidth=1)
    ax2.set_title(f"Grade {g} — Graded Activation Steering")
    ax2.set_xlim(lim_min_st, lim_max_st)
    ax2.set_ylim(lim_min_st, lim_max_st)
    ax2.set_xlabel("")
    ax2.set_ylabel("")

# Label
fig.suptitle("", fontsize=18)
fig.text(0.5, 0.01, "Pragmatic similarity", ha="center", fontsize=12)
fig.text(0.04, 0.5, "Logical similarity", va="center", rotation="vertical", fontsize=12)

### run ###
plt.tight_layout(rect=[0.05, 0.03, 1, 0.93])
plt.show()




# ------------------------------------------------------------
# Visualization (Histogram)
# ------------------------------------------------------------

def plot_delta_hist(csv_path, title):
    df = pd.read_csv(csv_path)
    df["lex_item"] = df["weak"] + "–" + df["strong"]

    base = df.groupby("lex_item")["pref_internal"].mean()
    steered = df.groupby("lex_item")["pref_steered"].mean()

    delta = (steered - base).dropna()

    plt.figure(figsize=(7, 4))
    sns.histplot(delta, bins=20, kde=True, color="#1f77b4")
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("")
    plt.ylabel("")
    plt.title(title)
    plt.tight_layout()
    plt.show()



### run ###
plot_delta_hist(
    "results_uniform.csv",
    "Δ Pragmatic intepretation rate (Uniform Activation Steering)"
)

plot_delta_hist(
    "results_grade.csv",
    "Δ Pragmatic intepretation rate (Graded Activation Steering)"
)

