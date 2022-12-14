library(FSA)# for vbFuns(), vbStarts(), confint.bootCase()
library(car)     # for Boot()
library(dplyr)   # for filter(), mutate()
library(ggplot2)

df = read.csv('leituras.csv')

# for(i in seq(0,10,0.1)){
#   line = c(NA,NA,i,"Jan",'S1',0)
#   df = rbind(df, line)
# }; rm(i,line)

df = df %>% mutate(age = as.numeric(age),
                   Lt = as.numeric(Lt))

vb = vbFuns(param = 'Typical')

f.starts = vbStarts(Lt ~ age, data = df)

f.fit = nls(Lt ~ vb(age, Linf, K, t0), data = df, start = f.starts)

predict2 = function(x) predict(x,data.frame(age=ages))

ages = seq(-1,12,by=0.2)
f.boot2 = Boot(f.fit, f=predict2)

preds1 = data.frame(ages,
                     predict(f.fit,data.frame(age=ages)),
                     confint(f.boot2))
names(preds1) = c("age","fit","LCI","UCI")

agesum = df %>% summarize(minage=min(age),maxage=max(age))
preds2 = filter(preds1,age>=agesum$minage,age<=agesum$maxage)

vbFitPlot <- ggplot() +
  geom_ribbon(data=preds2,aes(x=age,ymin=LCI,ymax=UCI),fill="gray90") +
  geom_point(data=df,aes(y=Lt,x=age),size=2,alpha=0.1) +
  geom_line(data=preds1,aes(y=fit,x=age),size=1,linetype=2) +
  geom_line(data=preds2,aes(y=fit,x=age),size=1) +
  scale_y_continuous(name="Total Length (mm)",limits=c(0,40),expand=c(0,0)) +
  scale_x_continuous(name="Age (years)",expand=c(0,0),
                     limits=c(-1,12),breaks=seq(0,12,2)) +
  theme_bw() +
  theme(panel.grid=element_blank())

makeVBEqnLabel <- function(fit) {
  # Isolate coefficients (and control decimals)
  cfs <- coef(fit)
  Linf <- formatC(cfs[["Linf"]],format="f",digits=1)
  K <- formatC(cfs[["K"]],format="f",digits=3)
  # Handle t0 differently because of minus in the equation
  t0 <- cfs[["t0"]]
  t0 <- paste0(ifelse(t0<0,"+","-"),formatC(abs(t0),format="f",digits=3))
  # Put together and return
  paste0("TL==",Linf,"~bgroup('(',1-e^{-",K,"~(age",t0,")},')')")
}
vbFitPlot + annotate(geom="text",label=makeVBEqnLabel(f.fit),parse=TRUE,
                     size=8,x=Inf,y=-Inf,hjust=1.1,vjust=-0.5)

