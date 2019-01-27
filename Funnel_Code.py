# 1

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def UserSim(n, lam):
    return list(np.random.exponential(1.0/lam, size=n))

# 1 (a)

print('1 (a)\n')
def funnel_viz(n, lam, bin_start = 0.0, bin_end = 3.0, stops = 13):
    users = UserSim(n, lam)
    hist = np.histogram(users, bins = list(np.linspace(bin_start, bin_end, num = stops, endpoint = True)))
    funnel = [1000]
    accum = 0
    for num in hist[0]:
        accum = accum+num
        funnel.append(1000-accum)
    plt.bar(x = list(hist[1]),height = funnel, width = 0.22)
    plt.title('Funnel Simulation with lambda = {}'.format(lam))
    plt.xlabel('Time (s)')
    plt.ylabel('Number of People staying in the funnel')
    plt.show()

funnel_viz(1000, 2, bin_start = 0.0, bin_end = 3.0, stops = 13)

# 1 (b)

print('1 (b)\n')
plt.figure(figsize=(10,6))
n = 1000, 
bin_start = 0.0
bin_end = 3.0
stops = 13
for lam in np.arange(0.2, 3.2, 0.2):
    users = UserSim(n, lam)
    hist = np.histogram(users, bins = list(np.linspace(bin_start, bin_end, num = stops, endpoint = True)))
    funnel = [1000]
    accum = 0
    for num in hist[0]:
        accum = accum+num
        funnel.append(1000-accum)
    plt.plot(list(hist[1]), funnel, label='lambda = {}'.format(lam), linewidth=0.5)
plt.title('Funnel Simulation')
plt.xlabel('Time (s)')
plt.ylabel('Number of People staying in the funnel')
plt.legend(bbox_to_anchor=(1, 1))
plt.show()


# 2

def EstLam1(users):
    return 1.0/np.mean(users)

# 2 (b)

print('2 (b)\n')
np.random.seed(42)
users = UserSim(1000, 1)
print('The estimated lambda is: %.5f' % EstLam1(users))

# 2 (c)

print('2 (c)\n')
def bootstraps_lambda(users, n=500, CI=0.95):
    estimates = []
    for i in range(n):
        sampled_users = np.random.choice(users, len(users))
        estimates.append(EstLam1(sampled_users))
    return (np.percentile(estimates, ((1-CI)/2)*100),
            np.percentile(estimates, (1-(1-CI)/2)*100),
            EstLam1(users))

np.random.seed(42)
lower, upper, estimate = bootstraps_lambda(users, n=500, CI=0.95)
print('The 95% confidence interval of lambda is ({}, {})'.format(np.round(lower, 5), np.round(upper, 5)))

# 2 (d)

print('2 (d)\n')
num_users = [100, 200, 500, 1000, 2000, 5000, 10000]
estimated_lambda = []
CI_lower = []
CI_upper = []
for j in num_users:
    np.random.seed(42)
    users = UserSim(j, 1)
    lower, upper, estimate = bootstraps_lambda(users, n=500, CI=0.95)
    estimated_lambda.append(estimate)
    CI_lower.append(lower)
    CI_upper.append(upper)

table_2_4 = pd.DataFrame({'Number of Users':num_users, 'Estimated Lambda':estimated_lambda, 'Lower CI':CI_lower, 'Upper CI':CI_upper})
table_2_4 = table_2_4[['Number of Users','Estimated Lambda','Lower CI','Upper CI']].round(5)

print table_2_4

plt.figure(figsize=(12,6))
plt.fill_between(table_2_4.loc[:,'Number of Users'],
                    table_2_4.loc[:,'Lower CI'],
                    table_2_4.loc[:,'Upper CI'],
                    color='b', alpha=0.3,
                    label='95% Confidence Interval')
plt.plot(table_2_4.loc[:,'Number of Users'],
         table_2_4.loc[:,'Estimated Lambda'],
         color='r')
plt.plot(num_users, np.ones(len(num_users)), 'b--', label='Real Lambda')
plt.legend()
plt.title('Estimated Lambda & CI with Different number of users')
plt.xlabel('Number of Users')
plt.ylabel('Estimated Lambda')
plt.show()

# 4

def HurdleFun(x, breaks):
    simulated_users = np.array(x)
    records = []
    for i in range(len(breaks)):
        if i == len(breaks)-1:   # append the last break
            records.append(simulated_users[simulated_users > breaks[i]])
        else:     # append in between breaks
            records.append(simulated_users[(simulated_users > breaks[i]) & (simulated_users <= breaks[i+1])])
    # insert the first break
    records.insert(0, simulated_users[simulated_users <= breaks[0]])
    counts = [len(l) for l in records]
    return counts

def EstLam2(hurdle_output, breaks):
    hurdle_output = np.array(hurdle_output)
    m0 = hurdle_output[0]
    m2 = hurdle_output[-1]
    
    def loglike(lam):
        ll = m0*np.log(1-np.exp(-lam*breaks[0])) + m2*(-lam*breaks[-1])
        if len(breaks) > 1:                       # deal with middle terms
            for i, m1 in enumerate(hurdle_output[1:-1]):
                ll = ll + m1*np.log(np.exp(-lam*breaks[i]) - np.exp(-lam*breaks[i+1]))
        return ll
    
    return loglike


def MaxMLE(hurdle_output, breaks, lam_list):
    max_log = -100000
    best_lam = 0
    PRT = EstLam2(hurdle_output, breaks)
    for i in lam_list:
        loglike = PRT(i)
        if loglike > max_log:
            max_log = loglike
            best_lam = i
    return best_lam


# 4 (a)
    
print('4(a)\n')

breaks = [0.25, 0.75]
np.random.seed(42)
lam_diff = []
for i in range(1000):
    users = UserSim(n=100, lam = 2)
    lam1 = EstLam1(users)
    lam2 = MaxMLE(HurdleFun(users, breaks), breaks = breaks, lam_list = list(np.arange(0.1,4,0.01)))
    lam_diff.append(lam1-lam2)
print('The average difference for breaks = %s is %.5f' % (breaks, np.mean(np.abs(lam_diff))))

breaks = [0.25, 3]
np.random.seed(42)
lam_diff = []
for i in range(1000):
    users = UserSim(n=100, lam = 2)
    lam1 = EstLam1(users)
    lam2 = MaxMLE(HurdleFun(users, breaks), breaks = breaks, lam_list = list(np.arange(0.1,4,0.01)))
    lam_diff.append(lam1-lam2)
print 'The average difference for breaks = %s is %.5f' % (breaks, np.mean(np.abs(lam_diff)))

breaks = [0.25, 10]
np.random.seed(42)
lam_diff = []
for i in range(1000):
    users = UserSim(n=100, lam = 2)
    lam1 = EstLam1(users)
    lam2 = MaxMLE(HurdleFun(users, breaks), breaks = breaks, lam_list = list(np.arange(0.1,4,0.01)))
    lam_diff.append(lam1-lam2)
print 'The average difference for breaks = %s is %.5f' % (breaks, np.mean(np.abs(lam_diff)))


# 4 (b)

print('4(b)\n')


for mylam in [1, 1.5, 2, 2.5, 3]:
    rec = []
    cs = []
    
    for c in np.arange(0.5,5,0.1):
        breaks = [0.25, c]
        np.random.seed(42)
        lam_diff = []
        for i in range(1000):
            users = UserSim(n=1000, lam = mylam)
            lam1 = EstLam1(users)
            lam2 = MaxMLE(HurdleFun(users, breaks), breaks = breaks, lam_list = list(np.arange(0.1,4,0.01)))
            lam_diff.append(lam1-lam2)
        rec.append(np.mean(np.abs(lam_diff)))
        cs.append(c)
    plt.plot(1 - np.exp(-mylam * np.array(cs)), rec, label = 'lambda = %0.2f' % mylam)
plt.title('Average difference vs Percentile')
plt.legend()
plt.xlabel('Corresponding percentile of the 2nd breakpoint')
plt.ylabel('Average difference')
plt.show()