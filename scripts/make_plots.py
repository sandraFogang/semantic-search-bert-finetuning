"""Génère 2 graphiques pour le README à partir des résultats d'hyperparamètres."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ============================================================================
# GRAPHIQUE 1 — Bar chart groupé : effet du learning rate et batch size
# ============================================================================

batch_sizes = ['bs = 16', 'bs = 32', 'bs = 64']
lr_data = {
    'lr = 1e-3': [43, 51, 57],
    'lr = 1e-4': [21, 22, 26],
    'lr = 1e-5': [10, 11, 14],
}
colors = ['#2E86AB', '#E07A5F', '#81B29A']

fig, ax = plt.subplots(figsize=(10, 6.2))
x = np.arange(len(batch_sizes))
width = 0.27

for i, (label, values) in enumerate(lr_data.items()):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, values, width, label=label, color=colors[i],
                  edgecolor='white', linewidth=1.5)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                f'{height}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#333')

ax.set_xlabel('Batch size', fontsize=13)
ax.set_ylabel('Top-10 precision (%)', fontsize=13)
ax.set_title('Effet des hyperparamètres sur la performance du retriever',
             fontsize=14, pad=18, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(batch_sizes, fontsize=12)
ax.set_ylim(0, 70)
ax.legend(loc='upper left', fontsize=11, framealpha=0.95, title='Learning rate',
          title_fontsize=11)
ax.grid(axis='y', alpha=0.25, linestyle='--')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#888')
ax.spines['bottom'].set_color('#888')

# Annotation du meilleur résultat
ax.annotate('Optimum', xy=(2 + width, 57.5),
            xytext=(1.5, 65),
            fontsize=11, color='#1a4d6a', fontweight='bold',
            ha='center',
            arrowprops=dict(arrowstyle='->', color='#1a4d6a', lw=1.5,
                            connectionstyle='arc3,rad=-0.2'))

plt.tight_layout()
plt.savefig('/home/claude/hyperparameter_barchart.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('✅ Graphique 1 sauvegardé : hyperparameter_barchart.png')


# ============================================================================
# GRAPHIQUE 2 — Scatter : corrélation entre convergence et performance
# ============================================================================

losses = [0.003, 0.004, 0.004, 0.040, 0.043, 0.046, 0.396, 0.401, 0.411]
top10 = [57, 51, 43, 26, 22, 21, 14, 11, 10]
lrs = ['1e-3', '1e-3', '1e-3', '1e-4', '1e-4', '1e-4', '1e-5', '1e-5', '1e-5']
bss = [64, 32, 16, 64, 32, 16, 64, 32, 16]

colors_map = {'1e-3': '#2E86AB', '1e-4': '#E07A5F', '1e-5': '#81B29A'}
sizes_map = {64: 280, 32: 170, 16: 90}

fig, ax = plt.subplots(figsize=(10, 6))

# Zones d'arrière-plan
ax.axvspan(0.001, 0.01, alpha=0.07, color='#2E86AB', zorder=0)
ax.axvspan(0.3, 0.5, alpha=0.07, color='#E07A5F', zorder=0)

# Scatter
for i in range(9):
    ax.scatter(losses[i], top10[i],
               s=sizes_map[bss[i]],
               c=colors_map[lrs[i]],
               edgecolors='white', linewidth=2,
               alpha=0.9, zorder=3)

# Annotations textuelles des zones
ax.text(0.0035, 4, 'Modèles\nconvergés', fontsize=10, color='#1a4d6a',
        ha='center', fontweight='bold', style='italic')
ax.text(0.39, 4, 'Modèles\nnon convergés', fontsize=10, color='#a04030',
        ha='center', fontweight='bold', style='italic')

# Légendes
lr_handles = [Line2D([0], [0], marker='o', color='w', label=f'lr = {lr}',
                     markerfacecolor=color, markersize=11, markeredgecolor='white',
                     markeredgewidth=1.5)
              for lr, color in colors_map.items()]
bs_handles = [Line2D([0], [0], marker='o', color='w', label=f'bs = {bs}',
                     markerfacecolor='#888', markersize=size_marker,
                     markeredgecolor='white', markeredgewidth=1.5)
              for bs, size_marker in zip([16, 32, 64], [6, 9, 13])]

leg1 = ax.legend(handles=lr_handles, loc='upper right', fontsize=10,
                 title='Learning rate', title_fontsize=10)
ax.add_artist(leg1)
ax.legend(handles=bs_handles, loc='upper right', fontsize=10,
          title='Batch size', title_fontsize=10,
          bbox_to_anchor=(1.0, 0.72))

ax.set_xscale('log')
ax.set_xlabel('Loss finale (échelle log)', fontsize=13)
ax.set_ylabel('Top-10 precision (%)', fontsize=13)
ax.set_title('Corrélation entre convergence du fine-tuning et performance',
             fontsize=14, pad=18, fontweight='bold')
ax.grid(True, alpha=0.25, linestyle='--')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#888')
ax.spines['bottom'].set_color('#888')
ax.set_ylim(-2, 70)

plt.tight_layout()
plt.savefig('/home/claude/loss_vs_top10.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('✅ Graphique 2 sauvegardé : loss_vs_top10.png')

print('\n📊 Les 2 graphiques sont prêts.')
