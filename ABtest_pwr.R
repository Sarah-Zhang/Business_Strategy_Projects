library(pwr)
# 3(c)
pwr.p.test(h=0.03, sig.level=0.001, power=0.8, alternative='two.sided')
# 3(e)
pwr.p.test(h=0.01, sig.level=0.001, power=0.5, alternative='two.sided')
