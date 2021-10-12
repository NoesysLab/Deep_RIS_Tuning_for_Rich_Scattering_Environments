import numpy as np
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

df = pd.DataFrame({
    'SNR'              : [10,         12.5,         15,          17.5,         20          ],
    'Exhaustive'       : [9.22139927, 9.99043500,   10.6061363,  11.118211980, 11.555723607],
    'Average'          : [0.85748744, 0.85910302,   0.85989377,  0.86065284,    0.861505496],
    'Top random'       : [8.60046115, 9.38711998,   9.96903115,  10.49563685,  10.710918369],
    'Deep RIS Setting': [8.95846785, 9.68480678,   10.2231392,  10.7734639,   11.116451725],
})

df['Average']                = df['Average'] * df['Exhaustive']
df['Average norm']           = df['Average'] / df['Exhaustive']
df['Top random norm']        = df['Top random'] / df['Exhaustive']
df['Deep RIS Setting norm'] = df['Deep RIS Setting'] / df['Exhaustive']


colors = sns.color_palette("Set1")
line_styles = ['-','--', ':', '-.', '.']
marker_styles = ["o", "s", 'x', 'D']
hatches = ['/', '-', '\\', '|', '+', 'x', 'o', 'O', '.', '*']

sns.set_style('ticks')
from matplotlib import rc
rc('text', usetex=True)
fontsize = 16
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize
plt.rcParams["savefig.dpi"] = 400
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.0
plt.rcParams["savefig.transparent"] = True
plt.rcParams["axes.labelsize"] = fontsize + 4
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}\usepackage{amsmath}'
#plt.rcParams['font.size'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize -2


plt.figure()
for i, name in enumerate(['Deep RIS Setting', 'Top random', 'Average', 'Exhaustive']):

    plt.plot(df['SNR'], df[name], c=colors[i], ls=line_styles[i], label=r"$\text{"+name+"}$", rasterized=False)
    plt.scatter(df['SNR'], df[name], c=colors[i], marker=marker_styles[i])

plt.legend()
plt.xlabel(r"${\rm SNR}\ ({\rm dB})$")
plt.ylabel(r"$\tilde{R}({\boldsymbol \varphi}) \ ({\rm bps/Hz})$")

plt.grid()
plt.savefig('./plots/paper_plots/achieved_rate_performances.pdf')
plt.show()



plt.figure()

ind = np.arange(len(df['SNR']))
plt.figure()
width = 0.3


for i, name in enumerate(['Deep RIS Setting', 'Top random', 'Average']):
    plt.bar(ind + i*width, df[name+" norm"], width, color=colors[i], hatch=hatches[i], label=r"$\text{"+name+"}$", rasterized=False)

plt.legend()
plt.xlabel(r"${\rm SNR}\ ({\rm dB})$")
plt.ylabel(r"$\tilde{R}({\boldsymbol \varphi}) \ {\rm [normalized]}$")
plt.ylim([0.8, 1.01])
plt.xticks(ind + width / 2, list(map(str, df['SNR'])))
plt.grid()
plt.savefig('./plots/paper_plots/achieved_rate_performances_normalized.pdf')
plt.show()







