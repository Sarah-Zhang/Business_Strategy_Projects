import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

# change filepath to your own
filepath = 'data/NueBevABTest.csv'

# load data
data = pd.read_csv(filepath, low_memory=False, parse_dates=['Customer Begin','Customer End'])
# truncate dates to months
data['begin_month'] = data['Customer Begin'].dt.to_period('M')
data['end_month'] = data['Customer End'].dt.to_period('M')

# Q1
print('\nQ1')
periods = pd.date_range('9/1/2016', periods=4, freq='M').to_period('M') # this is the testing periods
print('Monthly Churn Rates:')
churn_rates = []
for p in periods:
    total = data['Client number'][(data['begin_month']<=p) & (data['end_month']>=p)].count() # total active customers current month
    churn = data['Client number'][data['Churned'] & (data['end_month']==p)].count() # churn next month
    rate = 100.0 * churn / total
    churn_rates.append(rate)
    print('%s:\t%.2f%%' % (p+1, rate))
print('Monthly Average Churn Rates:\n%.2f%%' % np.mean(churn_rates))
overall = 100.0*data['Churned'].sum()/data.shape[0]
print('Overall Churn Rate:\n%.2f%%' % overall)

# Q7
print('\nQ7')
counts = data['Churned'].groupby(data['Margin Group']).agg(['count','sum']) # compute total / churn for each Margin Group
def p_val(counts):
    '''returns the p-value of the proportion test
    '''
    p15 = counts['sum']['Low']/counts['count']['Low']
    p18 = counts['sum']['High']/counts['count']['High']
    n15 = counts['count']['Low']
    n18 = counts['count']['High']
    Z = (p15-p18) / np.sqrt( p15*(1-p15)/n15 + p18*(1-p18)/n18 )
    return norm.cdf(Z)
print('p-value for proportaions test:\t%.4f' % p_val(counts))

# group by Account Size
group_counts = data['Churned'].groupby([data['Account Size'], data['Margin Group']]).agg(['count','sum'])
for s in sorted(data['Account Size'].unique()):
    pval = p_val(group_counts.loc[s])
    print('Account Size: %s\t p-value:\t%.4f' % (s, pval))

# Q8
print('\nQ8')
def power(delta, counts=counts):
    '''returns power of test given effect size and data
    Alternative: p2 = p1 + delta
    Reject Null if Z >= CO (= upper alpha=0.05 normal quantile)
    Power =  PR(Z>=CO|Alternative)
    '''
    CO = norm.isf(0.05)
    p15 = counts['sum']['Low']/counts['count']['Low']
    p18 = counts['sum']['High']/counts['count']['High']
    n15 = counts['count']['Low']
    n18 = counts['count']['High']
    Z = CO - delta / np.sqrt( p15*(1-p15)/n15 + p18*(1-p18)/n18 )
    return 1-norm.cdf(Z)
effect_sizes = np.arange(start=0.00, stop=0.20, step=0.01)
powers = np.array([power(e) for e in effect_sizes])

plt.plot(effect_sizes, powers)
plt.xlabel('effect size')
plt.ylabel('power')
plt.title('Power of tests with different Effect Sizes')
plt.show()
