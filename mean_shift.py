import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
veri, _ = make_blobs(n_samples=300, centers=3, random_state=42)
mean_shift = MeanShift(bandwidth=2)
mean_shift.fit(veri)
kume_merkezleri = mean_shift.cluster_centers_
etiketler = mean_shift.labels_
plt.figure(figsize=(8, 6))
plt.scatter(veri[:, 0], veri[:, 1], c=etiketler, cmap='viridis', s=50, alpha=0.7)
plt.scatter(kume_merkezleri[:, 0], kume_merkezleri[:, 1], color='red', marker='X', s=200, label='Küme Merkezleri')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.title('Mean-Shift Kümeleme')
plt.legend()
plt.show()