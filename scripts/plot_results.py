"""Plot completed paired evaluation counts; never substitutes loss for success."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
work = ROOT / 'results'
report = json.loads((work / 'paired_summary.json').read_text())
if report['status'] != 'completed':
    raise ValueError('Finish all planned evaluation episodes first.')
groups = ['original_range', 'wider_range']
counts = {label: [report['reports'][label][g]['successes'] for g in groups]
          for label in ('baseline', 'adapted')}
totals = [len(report['reports']['baseline'][g]['episodes']) for g in groups]
for label in counts:
    counts[label].append(sum(counts[label]))
totals.append(sum(totals))

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(9.4, 5.6))
x = np.arange(3)
for label, offset, color, title in [
    ('baseline', -.18, '#64748b', 'Saved model: 5,000 updates'),
    ('adapted', .18, '#007f87', 'Expanded data + 1,000 updates')]:
    values = np.array(counts[label]) / totals * 100
    bars = ax.bar(x + offset, values, width=.34, color=color, label=title)
    for bar, count, total in zip(bars, counts[label], totals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{count}/{total}', ha='center', weight='bold')
ax.set_xticks(x, ['Original start range\n10 paired trials',
                  'Wider start range\n10 paired trials', 'Combined\n20 paired trials'])
ax.set_ylabel('Successful pick-and-place (%)')
ax.set_ylim(0, 112)
ax.set_yticks(range(0, 101, 20))
ax.grid(axis='y', alpha=.2)
ax.set_axisbelow(True)
ax.legend(loc='upper left', bbox_to_anchor=(0, 1.18), frameon=False, ncol=2, fontsize=10)
fig.suptitle('Phone to Panda: paired simulation evaluation', x=.12, ha='left', weight='bold')
fig.text(.12, .03, 'Same five human recordings; learned movement/gripper with fixed wrist feedback.\n'
         'Additional data and training budget both change. Small custom-scene test, not a LIBERO benchmark.',
         fontsize=9, color='#475569')
fig.subplots_adjust(left=.12, right=.97, top=.80, bottom=.20)
output = work / 'paired_results.png'
fig.savefig(output, dpi=160)
print(output)
