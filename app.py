# app.py
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.metrics import roc_auc_score
import streamlit as st

plt.rcParams.update({
    "mathtext.fontset": "dejavusans",   # use matplotlib's mathtext font
    "font.family": "DejaVu Sans",
    "font.size": 12
})


# reproducibility helper
DEFAULT_SEED = 0

def bounded_normal(mean, std, size, rng):
    return np.clip(rng.normal(mean, std, size), 0, 1)


def compute_aucs(mu0_A, sigma0_A, mu1_A, sigma1_A,
                 mu0_B, sigma0_B, mu1_B, sigma1_B,
                 proportion_A=0.5, size=100000,
                 prevalence_A=0.5, prevalence_B=0.5,
                 compute_prf=False, compute_sauc=False,
                 seed=None):
    rng = np.random.default_rng(seed)

    n_A = int(size * proportion_A)
    n_B = size - n_A

    n1_A = int(n_A * prevalence_A)
    n0_A = n_A - n1_A

    n1_B = int(n_B * prevalence_B)
    n0_B = n_B - n1_B

    # draw samples
    class_0_A = bounded_normal(mu0_A, sigma0_A, n0_A, rng)
    class_1_A = bounded_normal(mu1_A, sigma1_A, n1_A, rng)

    class_0_B = bounded_normal(mu0_B, sigma0_B, n0_B, rng)
    class_1_B = bounded_normal(mu1_B, sigma1_B, n1_B, rng)

    # within-group AUCs
    y_A = np.concatenate([np.zeros(n0_A), np.ones(n1_A)])
    scores_A = np.concatenate([class_0_A, class_1_A])
    auc_A = roc_auc_score(y_A, scores_A) if len(np.unique(y_A)) > 1 else np.nan

    y_B = np.concatenate([np.zeros(n0_B), np.ones(n1_B)])
    scores_B = np.concatenate([class_0_B, class_1_B])
    auc_B = roc_auc_score(y_B, scores_B) if len(np.unique(y_B)) > 1 else np.nan

    # cross A/B:
    y_AB = np.concatenate([np.ones(n1_A), np.zeros(n0_B)])   # positives = class_1_A, negatives = class_0_B
    scores_AB = np.concatenate([class_1_A, class_0_B])
    auc_AB = roc_auc_score(y_AB, scores_AB) if len(np.unique(y_AB)) > 1 else np.nan

    y_BA = np.concatenate([np.ones(n1_B), np.zeros(n0_A)])   # positives = class_1_B, negatives = class_0_A
    scores_BA = np.concatenate([class_1_B, class_0_A])
    auc_BA = roc_auc_score(y_BA, scores_BA) if len(np.unique(y_BA)) > 1 else np.nan

    # overall AUC (positives = all class_1, negatives = all class_0)
    class_0_overall = np.concatenate([class_0_A, class_0_B])
    class_1_overall = np.concatenate([class_1_A, class_1_B])
    y_overall = np.concatenate([np.zeros_like(class_0_overall), np.ones_like(class_1_overall)])
    scores_overall = np.concatenate([class_0_overall, class_1_overall])
    auc_overall = roc_auc_score(y_overall, scores_overall) if len(np.unique(y_overall)) > 1 else np.nan

    auc_matrix = np.array([[auc_A, auc_AB],
                           [auc_BA, auc_B]])

    count_matrix = np.array([[n_A, n1_A + n0_B],
                             [n1_B + n0_A, n_B]])

    df = pd.DataFrame({
        'outcome': np.concatenate([np.zeros_like(class_0_A), np.ones_like(class_1_A),
                                   np.zeros_like(class_0_B), np.ones_like(class_1_B)]),
        'true_probs': np.concatenate([class_0_A, class_1_A, class_0_B, class_1_B]),
        'group': ['A'] * (n0_A + n1_A) + ['B'] * (n0_B + n1_B)
    })

    # optional extra metrics
    metrics = {}
    if compute_prf:

        y_A = np.concatenate([np.zeros(n0_A), np.zeros(n0_B), np.ones(n1_A)])
        scores_A = np.concatenate([class_0_A, class_0_B, class_1_A])
        prf_A = roc_auc_score(y_A, scores_A) if len(np.unique(y_A)) > 1 else np.nan

        y_B = np.concatenate([np.zeros(n0_B), np.zeros(n0_A), np.ones(n1_B)])
        scores_B = np.concatenate([class_0_B, class_0_A, class_1_B])
        prf_B = roc_auc_score(y_B, scores_B) if len(np.unique(y_B)) > 1 else np.nan

        metrics['PRF_A'] = prf_A
        metrics['PRF_B'] = prf_B

    if compute_sauc:

        y_A = np.concatenate([np.ones(n1_A), np.ones(n1_B), np.zeros(n0_A)])
        scores_A = np.concatenate([class_1_A, class_1_B, class_0_A])
        sAUC_A = roc_auc_score(y_A, scores_A) if len(np.unique(y_A)) > 1 else np.nan

        y_B = np.concatenate([np.ones(n1_B), np.ones(n1_A), np.zeros(n0_B)])
        scores_B = np.concatenate([class_1_B, class_1_A, class_0_B])
        sAUC_B = roc_auc_score(y_B, scores_B) if len(np.unique(y_B)) > 1 else np.nan

        metrics['sAUC_A'] = sAUC_A
        metrics['sAUC_B'] = sAUC_B

    metrics['AUC_overall'] = auc_overall

    return auc_matrix, auc_overall, df, count_matrix, metrics


def make_figure(auc_matrix, auc_overall, df, count_matrix, metrics=None,
                group_labels=None, cmap_min_clip=0.25, alpha=1,
                blue_color='#8ADCFB', red_color='#F6867A',
                show_counts=True, show_metrics=True):
    """
    Returns a matplotlib.figure.Figure containing the same two panels.
    """
    if group_labels is None:
        group_labels = ['A', 'B']

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    base_cmap = plt.colormaps.get_cmap('Blues')
    clamped_cmap = colors.ListedColormap(base_cmap(np.linspace(cmap_min_clip, 1.0, 256)))

    im = axs[0].imshow(auc_matrix, vmin=0.5, vmax=1.0, cmap=clamped_cmap, aspect='equal')
    axs[0].set_xlim(-0.5, 1.5)
    axs[0].set_ylim(1.5, -0.5)

    labels_tex = [[r"$\mathrm{AUC}_A$", r"$\mathrm{xAUC}_{A,B}$"],
                  [r"$\mathrm{xAUC}_{B,A}$", r"$\mathrm{AUC}_B$"]]

    for (i, j), val in np.ndenumerate(auc_matrix):
        auc_text = f"{val:.2f}" if not np.isnan(val) else "nan"
        if show_counts:
            N = int(count_matrix[i, j])
            # keep newline; mathtext inside f-string is fine
            txt = f"{labels_tex[i][j]}\n{auc_text}\n(n = {N:,.0f})"
        else:
            txt = f"{labels_tex[i][j]}\n{auc_text}"
        axs[0].text(j, i, txt, ha='center', va='center', fontsize=12, color='white')


    axs[0].set_xticks([0, 1])
    axs[0].set_yticks([0, 1])
    axs[0].set_xticklabels(group_labels, fontsize=11)
    axs[0].set_yticklabels(group_labels, fontsize=11)
    total_n = int(count_matrix.sum())
    axs[0].set_title(f"AUC Matrix\n")

    if show_metrics and (metrics is not None):
        bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", alpha=0.5)
        if 'PRF_A' in metrics:
            axs[0].text(0.5, 0.0, f"{r"$\mathrm{PRF_A}$"}\n{metrics['PRF_A']:.2f}", ha='center', va='center', fontsize=10, color='black', bbox=bbox_props)
        if 'PRF_B' in metrics:
            axs[0].text(0.5, 1.0, f"{r"$\mathrm{PRF_B}$"}\n{metrics['PRF_B']:.2f}", ha='center', va='center', fontsize=10, color='black', bbox=bbox_props)
        if 'sAUC_A' in metrics:
            axs[0].text(0.0, 0.5, f"{r"$\mathrm{sAUC_A}$"}\n{metrics['sAUC_A']:.2f}", ha='center', va='center', fontsize=10, color='black', bbox=bbox_props)
        if 'sAUC_B' in metrics:
            axs[0].text(1.0, 0.5, f"{r"$\mathrm{sAUC_B}$"}\n{metrics['sAUC_B']:.2f}", ha='center', va='center', fontsize=10, color='black', bbox=bbox_props)
        axs[0].text(0.5, 0.5, f"{r"$\mathrm{AUC_{overall}}$"}\n{metrics.get('AUC_overall', np.nan):.2f}", ha='center', va='center', fontsize=10, color='black', bbox=bbox_props)

    # Right: violin
    palette = {0: blue_color, 1: red_color}
    sns.violinplot(
        data=df,
        hue='outcome',
        y='true_probs',
        x='group',
        inner='quartile',
        palette=palette,
        ax=axs[1],
        cut=0,
        split=True
    )

    # clean up
    for coll in axs[1].collections:
        try:
            coll.set_edgecolor('none')
        except Exception:
            pass
        try:
            coll.set_alpha(alpha)
        except Exception:
            pass
    for line in axs[1].lines:
        try:
            line.set_color('white')
            line.set_linewidth(1.5)
        except Exception:
            pass

    handles, labels = axs[1].get_legend_handles_labels()
    legend = axs[1].legend(handles=handles, labels=['0', '1'], title='Label', loc='upper right')

    # Remove edges from legend patches
    for patch in legend.get_patches():
        patch.set_edgecolor('none')

    axs[1].set_title('Model Outputs by Subgroup\n')
    axs[1].set_xlabel('Subgroup')
    axs[1].set_ylabel(r"Simulated Estimated Probability, $\hat{p}$" + "\n")
    axs[1].set_xticks([0, 1])
    axs[1].set_xticklabels(group_labels)
    axs[1].grid(True, linestyle='--', alpha=0.3)

    for ax in axs:
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    return fig


# Streamlit UI
st.set_page_config(layout="wide", page_title="AUC dashboard")
st.title("AUC under mixture of populations")

with st.sidebar:
    st.header("Simulation Parameters")
    seed = st.number_input("Random Seed", value=DEFAULT_SEED, step=1)
    size = st.number_input("Total Sample Size", value=100000, step=1000)
    st.header("Case-mix Settings")
    proportion_A = st.slider(r"$p^A$ Proportion of Subgroup A", 0.01, 0.99, 0.3, 0.01)
    prevalence_A = st.slider(r"$\pi^A$ Outcome Prevalence in A", 0.0, 1.0, 0.5, 0.01)
    prevalence_B = st.slider(r"$\pi^B$ Outcome Prevalence in B", 0.0, 1.0, 0.5, 0.01)
    st.header("Simulation Means")
    mu0_A = st.slider(r"$\mu_0^A$ Mean $\hat{p}$ for negative-samples in A", 0.0, 1.0, 0.30, 0.01)
    mu1_A = st.slider(r"$\mu_1^A$ Mean $\hat{p}$ for positive-samples in A", 0.0, 1.0, 0.45, 0.01)
    mu0_B = st.slider(r"$\mu_0^B$ Mean $\hat{p}$ for negative-samples in B", 0.0, 1.0, 0.50, 0.01)
    mu1_B = st.slider(r"$\mu_1^B$ Mean $\hat{p}$ for positive-samples in B", 0.0, 1.0, 0.65, 0.01)
    st.header("Simulation Standard Deviations")
    sigma0_A = st.slider(r"$\sigma_0^A$ Std $\hat{p}$ for negative-samples in A", 0.01, 0.5, 0.10, 0.01)
    sigma1_A = st.slider(r"$\sigma_1^A$ Std $\hat{p}$ for positive-samples in A", 0.01, 0.5, 0.10, 0.01)
    sigma0_B = st.slider(r"$\sigma_0^B$ Std $\hat{p}$ for negative-samples in B", 0.01, 0.5, 0.10, 0.01)
    sigma1_B = st.slider(r"$\sigma_1^B$ Std $\hat{p}$ for positive-samples in B", 0.01, 0.5, 0.10, 0.01)

    st.header("Display Options")
    compute_prf = st.checkbox("Compute PRF metrics", value=False)
    compute_sauc = st.checkbox("Compute sAUC metrics", value=False)
    show_counts = st.checkbox("Show counts on heatmap", value=False)

# compute
auc_matrix, auc_overall, df, count_matrix, metrics = compute_aucs(
    mu0_A, sigma0_A, mu1_A, sigma1_A,
    mu0_B, sigma0_B, mu1_B, sigma1_B,
    proportion_A=proportion_A,
    prevalence_A=prevalence_A,
    prevalence_B=prevalence_B,
    compute_prf=compute_prf,
    compute_sauc=compute_sauc,
    seed=seed
)

fig = make_figure(auc_matrix, auc_overall, df, count_matrix, metrics=metrics,
                  show_counts=show_counts, show_metrics=True)


st.pyplot(fig)

# vertical space
st.markdown("###")

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
buf.seek(0)
st.download_button(
    label="Download PNG",
    data=buf.getvalue(),
    file_name="auc_matrix.png",
    mime="image/png", #centred
    use_container_width=True, #but smaller
)
