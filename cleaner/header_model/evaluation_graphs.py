import matplotlib.pyplot as plt
import numpy as np

# === METRICHE ===
dev = {"Precision": 0.905, "Recall": 0.929, "F1": 0.917}
test = {"Precision": 0.908, "Recall": 0.866, "F1": 0.887}

labels = list(dev.keys())
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(6, 4))

rects1 = ax.bar(x - width/2, list(dev.values()), width, label='Dev set')
rects2 = ax.bar(x + width/2, list(test.values()), width, label='Test set')

ax.set_ylabel('Score')
ax.set_ylim(0, 1)
ax.set_title('Confronto metriche modello spaCy (HEADER)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.bar_label(rects1, fmt='%.3f', padding=3)
ax.bar_label(rects2, fmt='%.3f', padding=3)

plt.tight_layout()

# Salva il grafico
plt.savefig("metrics_comparison.png", dpi=300)  # PNG ad alta risoluzione
# plt.savefig("metrics_comparison.pdf")         # in PDF
# plt.savefig("metrics_comparison.svg")         # in SVG (vettoriale)