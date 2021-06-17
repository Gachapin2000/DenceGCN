import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.construct import rand
import seaborn as sns

for file in glob.glob('./result/*.npy'):
    plt.figure()
    att = np.load(file)
    sns.heatmap(att, vmin=0., vmax=1., center=0.5, cmap='Reds')
    plt.savefig(os.path.splitext(file)[0] + '.png')
    plt.close()
