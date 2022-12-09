library(FSA)# for vbFuns(), vbStarts(), confint.bootCase()
library(car)     # for Boot()
library(dplyr)   # for filter(), mutate()
library(ggplot2)

df = read.csv('leituras.csv')

names(df)

for(i in seq(0,10,0.1)){
  line = c(NA,NA,i,"Jan",'S1',0)
  df = rbind(df, line)
}; rm(i,line)

df = df %>% mutate(age = as.numeric(age),
                   Lt = as.numeric(Lt))

vb = vbFuns(param = 'Typical')

f.starts = vbStarts(Lt ~ age, data = df)
f.starts$Linf = 30; f.starts$K = 1; f.starts$t0 = 0

f.fit = nls(Lt ~ vb(age, Linf, K, t0), data = df, start = f.starts)

f.fit

df %>%
ggplot() +
  geom_point(aes(x = age, y = Lt))
