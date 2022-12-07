library(FSA)# for vbFuns(), vbStarts(), confint.bootCase()
library(car)     # for Boot()
library(dplyr)   # for filter(), mutate()
library(ggplot2)

df = read.csv('leituras.csv')

vb = vbFuns(param = 'Typical')

f.starts = vbStarts(Lt ~ age, data = df)

f.fit = nls(Lt ~ vb(age, Linf, K, t0), data = df, start = f.starts)

f.fit