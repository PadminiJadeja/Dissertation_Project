import numpy as np
from sklearn import metrics
import pandas as pd
import sys
import matplotlib.pyplot as plt
import glob
from scipy.optimize import brentq
from scipy.interpolate import interp1d
plt.rcParams.update({'figure.max_open_warning': 0})
import warnings
warnings.filterwarnings("ignore")


def check(i,anomalous_frames):
    for x in anomalous_frames:
        # print(x)
        if in_range(i,x):
            return True
    return False

def in_range(i,x):
    # print(i)
    # check if frame (i) is present in the anomalous frame range or not
    return int(x[0]) <= i <= int(x[1])

output_files = sorted(glob.glob('*.csv'))
# print(output_files)
sum_auc = 0
sum_eer = 0

# frames_ShanghaiTech contains the ground truth for each test video in ShanghaiTech dataset separated by new line
with open('frames_ShanghaiTech') as f:
    lines = [line.rstrip('\n') for line in f]

# fetching the frame ranges
rang=[x.split(',') for x in lines]

#Frame ranges in frames_ShanghaiTech is stored in form of a:b, so
#splitting by : and storing in a list
value = list(map(lambda xs: [x.split(':') for x in xs],rang))

# results.txt will contain the auc and eer values of each test videos
with open('results.txt','w') as f:
    for file_num,_file in enumerate(output_files):
        data = pd.read_csv(_file)


        scores = np.array(data['score'])

        anomalous_frames = value[file_num]
        true_values = []

        for i in range(len(scores)):
            j= i+1
            if check(j,anomalous_frames):
                # if frame j is present in anomaly frame ranges then append 0 to it's ground truth value as we are comparing the 
                # regularity score. If a frame is anomalous it's regularity score will be 0.
                true_values.append(0)
            else:
                true_values.append(1)

        y_true = np.array(true_values)

        # roc
        fpr,tpr,thresholds = metrics.roc_curve(y_true,scores)

        # eer
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        try:
            # auc
            auc = metrics.roc_auc_score(y_true,scores)
        except ValueError:
            pass
        
        f.write('auc = ' + str(auc) + ', eer = ' + str(eer) + '\n')
        sum_auc += auc
        sum_eer += eer


        plt.figure()
        plt.plot(fpr,tpr)
        
        n = file_num+1 
        plt.savefig(f"ROC/roc_{n}.pdf")

# overall auc and eer values
print(sum_auc/len(output_files))
print(sum_eer/len(output_files))