import numpy as np
from collections import Counter


################################################################################
#Group Sprinkle
#Member:
#    Chenxi Ge
#    Yue Lan
#    Dixin Yan
#    Xiaowen (Sarah) Zhang
################################################################################


##########
# 1. 
def trucksim(n, d):
    return np.random.choice(d.keys(), n, p = d.values())

d = {'<10':0.08, '10-15':0.27, '15-20':0.10, '20-25':0.11,
     '25-30':0.15, '30-35':0.20, '35-37':0.07, '37+':0.02}


##########
# 2.
print 'Question 2'

def truck1CI(keyname, simlist, n, alpha):
    perc_total = []
    for i in range(n):
        sample_list = np.random.choice(simlist, len(simlist))
        percentage = 1.0 * Counter(sample_list)[keyname]/ len(sample_list)
        perc_total.append(percentage)
    return (np.mean(perc_total),
            np.percentile(perc_total, 100*(1-alpha/2)),
            np.percentile(perc_total, 100*(alpha/2)),
            np.std(perc_total)/np.sqrt(len(perc_total)))


np.random.seed(42)
s1 = trucksim(1000, d)

print truck1CI('30-35', s1, 100, 0.1)
print truck1CI('30-35', s1, 1000, 0.1)
print truck1CI('30-35', s1, 2500, 0.1)
print truck1CI('30-35', s1, 5000, 0.1)

print truck1CI('<10', s1, 100, 0.1)
print truck1CI('<10', s1, 1000, 0.1)
print truck1CI('<10', s1, 2500, 0.1)
print truck1CI('<10', s1, 5000, 0.1)


##########
# 3.
print 'Question 3'

a = {'<10':452, '10-15':1212, '15-20':625, '20-25':653,
     '25-30':713, '30-35':858, '35-37':368, '37+':108}

np.random.seed(42)
s2 = trucksim(4989, d)

print truck1CI('35-37', s2, 200, 0.05)
print truck1CI('37+', s2, 200, 0.05)
print truck1CI('35-37', s2, 200, 0.1)
print truck1CI('37+', s2, 200, 0.1)


##########
# 4.
print 'Question 4'

np.random.seed(42)
cnt1 = []
for i in range(1000):
    simulated = trucksim(4989, d)
    target_count = Counter(simulated)['35-37']
    cnt1.append(target_count >= 368)
print np.mean(cnt1)


np.random.seed(42)
cnt2 = []
for i in range(1000):
    simulated = trucksim(4989, d)
    target_count = Counter(simulated)['37+']
    cnt2.append(target_count >= 108)
print np.mean(cnt2)


##########
# 5.
print 'Question 5'

np.random.seed(42)
cnt3 = []
for i in range(1000):
    simulated = trucksim(4989, d)
    target_count1 = Counter(simulated)['35-37']
    target_count2 = Counter(simulated)['37+']
    cnt3.append((target_count1 >= 368) and (target_count2 >= 108))
print np.mean(cnt3)


