import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import itertools
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error


ground_truth1 = np.array([0.04508128, 0.01565821, 0.05240657, 0.12462071, 0.05920897,
       0.10597309, 0.12528312, 0.1101626 , 0.35545434, 0.22989773,
       0.25730771, 0.17682765, 0.07371242, 0.25837516, 0.20530505,
       0.07768637, 0.26387247, 0.04608677, 0.04229629, 0.0979792 ,
       0.04848155, 0.0688698 , 0.03725882, 0.057577  , 0.08134818,
       0.06010612, 0.06074399, 0.04146915, 0.15592438, 0.0728235 ])
predicted1 =   np.array([0.04153217, 0.01714088, 0.06103546, 0.15621424, 0.0612621 ,
       0.10838877, 0.13287678, 0.11641593, 0.33920664, 0.22086348,
       0.24518429, 0.19168423, 0.07717307, 0.24465008, 0.1789077 ,
       0.08101166, 0.24474685, 0.04177524, 0.04284425, 0.09129904,
       0.04860585, 0.079511  , 0.03612208, 0.06090396, 0.08253996,
       0.05863236, 0.07378218, 0.04197218, 0.1618197 , 0.07151713])




ground_truth2 = np.array([0.03743646, 0.01666614, 0.0681466 , 0.15588258, 0.0630194 ,
       0.1065863 , 0.1327078 , 0.11949003, 0.34968167, 0.1918563 ,
       0.25989408, 0.19924516, 0.07973058, 0.2390759 , 0.17330038,
       0.08765843, 0.25926002, 0.0477247 , 0.04208124, 0.08321953,
       0.04633852, 0.08201241, 0.03350071, 0.05581625, 0.0934853 ,
       0.05964909, 0.07249699, 0.04353695, 0.1664826 , 0.06290133])
predicted2 = np.array([0.04477379, 0.01517053, 0.0604268 , 0.14671998, 0.06538912,
       0.11272673, 0.12886505, 0.11337342, 0.3495872 , 0.21676657,
       0.2404934 , 0.1950069 , 0.07661301, 0.24831   , 0.17819524,
       0.07772222, 0.23690067, 0.04480321, 0.04515862, 0.09132396,
       0.0507038 , 0.07198611, 0.03407321, 0.0577314 , 0.08658546,
       0.05812396, 0.07101607, 0.04201821, 0.15345624, 0.07160225],)








colors = sns.color_palette("Set1")
sns.set_style('ticks')
f_range = np.arange(1, len(ground_truth1) + 1)
zeros = np.zeros(f_range.max())
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
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
#plt.rcParams['font.size'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize -2



fig_i = 0
for ground_truth, predicted in [(ground_truth1, predicted1), (ground_truth2, predicted2)]:

       plt.figure()
       plt.plot(f_range, ground_truth, c=colors[0], ls='-', label='Ground truth')
       plt.scatter(f_range, ground_truth, c=colors[0])
       plt.fill_between(f_range, ground_truth, zeros, alpha=0.5, color=colors[0])
       plt.plot(f_range, predicted, c=colors[1], ls=':', label='Predicted')
       plt.scatter(f_range, predicted, c=colors[1])
       plt.fill_between(f_range, predicted, zeros, alpha=0.5, color=colors[1])
       plt.xlim([1, f_range.max()])
       plt.ylim([0, 0.405])
       plt.xlabel('$f$')
       plt.ylabel('$\mathbb{E}\{|H(f)|^2\}$')

       if fig_i == 0:
              plt.legend()
       fig_i += 1

       plt.grid()
       plt.savefig(f'./plots/paper_plots/network_magnitudes_prediction_{fig_i}.pdf')
       plt.show()

