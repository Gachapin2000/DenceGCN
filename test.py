import glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from data import Planetoid

'''for file in glob.glob('./result/*.npy'):
    plt.figure()
    att = np.load(file)
    sns.heatmap(att, vmin=0., vmax=1., center=0.4, cmap='Reds')
    plt.savefig(os.path.splitext(file)[0] + '.png')
    plt.close()'''


root = './data/{}_{}'.format(config['dataset'], config['pre_transform'])
dataset = Planetoid(root          = root.lower(),
                    name          = config['dataset'], 
                    split         = config['split'], 
                    transform     = eval(config['transform']),
                    pre_transform = eval(config['pre_transform']))
data = dataset[0].to(device)