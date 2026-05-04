"""
Manually aggregate HGT results across seeds and compute mean ± std.
"""

import numpy as np

# i paste the result of each seed mamnually because the cluster has oom issue and every time i only have enough
# time and space to run one or two seed
seed_results = [
    # seed 0
    {'Recall@1': 0.1735, 'Recall@5': 0.3608, 'Recall@10': 0.4616,
     'Mean Haversine (km)': 2.071, 'Med Haversine (km)': 0.987},
    # seed 1
    {'Recall@1': 0.1751, 'Recall@5': 0.3629, 'Recall@10': 0.4638,
     'Mean Haversine (km)': 2.047, 'Med Haversine (km)': 0.971},
    # seed 2
    {'Recall@1': 0.1744, 'Recall@5': 0.3618, 'Recall@10': 0.4627,
     'Mean Haversine (km)': 2.059, 'Med Haversine (km)': 0.979},
]

metrics = ['Recall@1', 'Recall@5', 'Recall@10', 'Mean Haversine (km)', 'Med Haversine (km)']

print('=== GRU + HGT — Seed Results ===\n')
for i, r in enumerate(seed_results):
    print(f'Seed {i}: ' + '  '.join(f'{m}={r[m]:.4f}' for m in metrics))

print('\n=== Mean ± Std across seeds ===\n')
for m in metrics:
    vals = [r[m] for r in seed_results]
    print(f'  {m:<25}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')

print('\n=== LaTeX Table Row ===\n')
row_vals = []
for m in metrics:
    vals = [r[m] for r in seed_results]
    row_vals.append(f'{np.mean(vals):.4f} $\\pm$ {np.std(vals):.4f}')
print('GRU + HGT (ours) & ' + ' & '.join(row_vals) + ' \\\\')
