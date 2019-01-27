import numpy as np
import pandas as pd
from scipy.stats import norm, t
import matplotlib.pyplot as plt
import math

# Q2
# change path here
path = 'data/'
data1 = pd.read_csv(path+'outputDataSheet1.csv', low_memory=False)
data2 = pd.read_csv(path+'outputDataSheet2.csv', low_memory=False)
# aggregate revenue data to customer level and merge with conversion data
data = data1.merge(data2.Cost.groupby([data2.OfferType,data2.UserNo]).sum().reset_index(), 
                   on=['OfferType','UserNo'], how='left')
data.fillna(0, inplace=True)

## 2(a)
def pval_2p2n(counts):
    p1 = 1.0*counts.iloc[0]['mean']
    p2 = 1.0*counts.iloc[1]['mean']
    n1 = 1.0*counts.iloc[0]['count']
    n2 = 1.0*counts.iloc[1]['count']
    Z = abs(p1-p2) / np.sqrt( p1*(1-p1)/n1 + p2*(1-p2)/n2 )
    return Z, 1-norm.cdf(Z)
counts = data.SuccessFlag.groupby(data.OfferType).agg(['mean','count'])
print('\nQ2(a)')
print('test statistics for proportaions test:\t%.4f \np-value for proportaions test:\t%.4f' 
      % pval_2p2n(counts))

## 2(b)
def pval_2mu2n(counts):
    mu1 = 1.0*counts.iloc[0]['mean']
    mu2 = 1.0*counts.iloc[1]['mean']
    n1 = 1.0*counts.iloc[0]['count']
    n2 = 1.0*counts.iloc[1]['count']
    var1 = 1.0*counts.iloc[0]['std']**2
    var2 = 1.0*counts.iloc[1]['std']**2
    stats = abs(mu1-mu2) / np.sqrt( var1/n1 + var2/n2 )
    df = min(n1-1, n2-1)
    return stats, df, 1-t.cdf(stats, df)
### test of paying customers only
counts = data[data.SuccessFlag==1].Cost.groupby(data.OfferType).agg(['count','mean','std'])
print('\nQ2(b).1')
print('test statistics for mean test with paying users:\t%.4f \ndegrees of freedom for mean test with paying users:\t%d \np-value for mean test with paying users:\t\t%.4f' 
      % pval_2mu2n(counts))
### test of all users
counts = data.Cost.groupby(data.OfferType).agg(['count','mean','std'])
print('\nQ2(b).2')
print('test statistics for mean test with all users:\t\t%.4f \ndegrees of freedom for mean test with all users:\t%d \np-value for mean test with all users:\t\t\t%.4f' 
      % pval_2mu2n(counts))

# Q3

## 3(a)
print('\nQ3(a)')
powers = []
co = norm.isf(0.025)
for n in range(390,400):
    p = 1 - norm.cdf(co+.2/np.sqrt(2.0/n)) + norm.cdf(-co+.2/np.sqrt(2.0/n))
    powers.append([n,p,abs(p-.8)])
print('number of users:\t%d' % sorted(powers, key=lambda x:x[2])[0][0])

## 3(b)
print('\nQ3(b)')
print('minimum sample size:\t%d' % np.ceil(-np.log2(.001/2)))

## 3(d)
print('\nQ3(d)')
print('minimum sample size:\t%d' % np.ceil(math.log(0.001, 0.05)))

# Q4

def genTwoBinomial(n, p):
    return np.random.binomial(n, p), np.random.binomial(n, p)

def oneSideTest(x1, x2, n1, n2, alpha):
    p1 = 1.0 * x1 / n1
    p2 = 1.0 * x2 / n2
    
    test_stat = (p1 - p2) / np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    return 1 - norm.cdf(test_stat) < alpha

def twoSideTest(x1, x2, n1, n2, alpha):
    p1 = 1.0 * x1 / n1
    p2 = 1.0 * x2 / n2
    
    p = 1.0 * (x1+x2) / (n1+n2)
    
    test_stat = (p1 - p2) / np.sqrt(p * (1-p) * (1.0/n1 + 1.0/n2))
    return (1 - norm.cdf(test_stat) < alpha/2.0) or (norm.cdf(test_stat) < alpha/2.0)

np.random.seed(42)


p_list = np.linspace(0.1, 0.9, num=18)
n_list = [100, 500, 1000, 5000, 10000]

print('\nQ4')

n_prob_list = []
for n in n_list:
    print('running simulation with n = %d' % n)
    prob_to_reject_list = []
    
    for p in p_list:
        repeat = 100000
        reject_list = []
        
        for _ in range(repeat):
            x1, x2 = genTwoBinomial(n, p)
            reject_list.append(int(oneSideTest(x1, x2, n, n, 0.1)))
        
        prob_to_reject_list.append(np.mean(reject_list))
    
    n_prob_list.append(prob_to_reject_list)
    
    
for probs in n_prob_list:
    plt.plot(p_list, probs)
plt.legend(n_list)
plt.xlabel('p')
plt.ylabel('error rate')
plt.title('Probability to Reject Null (One tailed)')
plt.show()




np.random.seed(42)

p_list = np.linspace(0.1, 0.9, num=18)
n_list = [100, 500, 1000, 5000, 10000]

n_prob_list = []
for n in n_list:
    print('running simulation with n = %d' % n)
    prob_to_reject_list = []
    
    for p in p_list:
        repeat = 100000
        reject_list = []
        
        for _ in range(repeat):
            x1, x2 = genTwoBinomial(n, p)
            reject_list.append(int(twoSideTest(x1, x2, n, n, 0.1)))
        
        prob_to_reject_list.append(np.mean(reject_list))
    
    n_prob_list.append(prob_to_reject_list)

for probs in n_prob_list:
    plt.plot(p_list, probs)
plt.legend(n_list)
plt.xlabel('p')
plt.ylabel('error rate')
plt.title('Probability to Reject Null (Two tailed)')
plt.show()    

