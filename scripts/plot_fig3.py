import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SEED = 20260605
np.random.seed(SEED)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "..", "outputs")
FIGURE_DIR = os.path.join(SCRIPT_DIR, "..", "manuscript", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

# Load the already-computed data so we don't need the Google Drive database!
stability_path = f"{OUTPUTS_DIR}/test_retest_stability.csv"
if not os.path.exists(stability_path):
    print(f"Error: Could not find {stability_path}. Please check your paths.")
    exit(1)

stability_df = pd.read_csv(stability_path)

def prettify_feature_label(name):
    label = str(name)
    label = label.replace('global_beta_alpha_ratio', 'Global | beta/alpha ratio')
    label = label.replace('occipital_theta_alpha_ratio', 'Occipital | theta/alpha ratio')
    label = label.replace('parietal_theta_alpha_ratio', 'Parietal | theta/alpha ratio')
    label = label.replace('global_theta_alpha_ratio', 'Global | theta/alpha ratio')
    label = label.replace('occipital_beta_alpha_ratio', 'Occipital | beta/alpha ratio')
    label = label.replace('parietal_beta_alpha_ratio', 'Parietal | beta/alpha ratio')
    label = label.replace('global_', 'Global | ')
    label = label.replace('frontal_', 'Frontal | ')
    label = label.replace('central_', 'Central | ')
    label = label.replace('parietal_', 'Parietal | ')
    label = label.replace('temporal_', 'Temporal | ')
    label = label.replace('occipital_', 'Occipital | ')
    label = label.replace('abs_', 'absolute ')
    label = label.replace('rel_', 'relative ')
    label = label.replace('high_alpha', 'high alpha')
    label = label.replace('low_alpha', 'low alpha')
    label = label.replace('_', ' ')
    label = ' '.join(label.split())
    return label[0].upper() + label[1:] if label else label

def categorize_feature(name):
    if 'ratio' in name: return 'Ratio'
    elif 'alpha' in name: return 'Alpha'
    elif 'beta' in name: return 'Beta'
    elif 'theta' in name: return 'Theta'
    elif 'delta' in name: return 'Delta'
    else: return 'Other'

exclude_terms = ['age', 'sex', 'session', 'ses', 'late', 'participant', 'filename', 'file', 'handedness', 'task', 'acquisition']

stability_plot_df = stability_df[
    ~stability_df['feature'].str.contains('|'.join(exclude_terms), case=False, regex=True, na=False)
].copy()

top_stability = stability_plot_df.sort_values('test_retest_rho', ascending=False).head(15).copy()
top_stability['display_label'] = top_stability['feature'].apply(prettify_feature_label)
top_stability['category'] = top_stability['feature'].apply(categorize_feature)

color_map = {
    'Ratio': '#D55E00', 'Alpha': '#0072B2', 'Beta': '#009E73',
    'Theta': '#CC79A7', 'Delta': '#F0E442', 'Other': '#999999'
}
top_stability['color'] = top_stability['category'].map(lambda x: color_map.get(x, '#999999'))

plot_labels = top_stability['display_label'][::-1].tolist()
plot_vals = top_stability['test_retest_rho'][::-1].tolist()
plot_colors = top_stability['color'][::-1].tolist()

fig, ax = plt.subplots(figsize=(8.5, 6))
bars = ax.barh(plot_labels, plot_vals, color=plot_colors, edgecolor='none', height=0.7)

for bar, val in zip(bars, plot_vals):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', ha='left', fontsize=9, color='black')

ax.set_xlabel('Test-retest Spearman rho', fontsize=11, fontweight='bold')
ax.set_ylabel('')
ax.set_xlim(0.70, 0.95)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', labelsize=10)

legend_handles = [mpatches.Patch(color=color_map[cat], label=cat) for cat in top_stability['category'].unique()]
ax.legend(handles=legend_handles, title='Feature Category', loc='lower right', frameon=False)

plt.tight_layout()
out_png = f"{FIGURE_DIR}/fig3_test_retest_stability.png"
out_pdf = f"{FIGURE_DIR}/fig3_test_retest_stability.pdf"
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
print(f"Saved {out_png}")
print(f"Saved {out_pdf}")
plt.show()
