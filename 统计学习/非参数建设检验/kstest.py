from statsmodels.sandbox.stats.runs import runstest_2samp
from statsmodels.stats.descriptivestats import sign_test

import scipy.stats as stats

# 分布的检验

from scipy.stats import kstest
import numpy as np
x = np.random.normal(0,1,1000)
test_stat = kstest(x, 'norm',args=(x.mean(),x.std()))
print(test_stat)

stats.anderson()
stats.shapiro()
stats.ranksums()
stats.mannwhitneyu()
stats.wilcoxon()
stats.ks_2samp()


runstest_2samp()