library(rugarch)

#tf is data
tf=read.delim("../data/ftse.txt", header = FALSE) # can similarly replace this with other datasets

mysize=nrow(tf);
train=nrow(tf)-130+1;

ll=0

specg = ugarchspec(variance.model = list(model="sGARCH"), mean.model = list(armaOrder = c(0,0), include.mean = FALSE)) ## GARCH model
#specg = ugarchspec(variance.model = list(model="gjrGARCH"), mean.model = list(armaOrder = c(0,0), include.mean = FALSE)) ## GJR model
#specg = ugarchspec(variance.model = list(model="EGARCH"), mean.model = list(armaOrder = c(0,0), include.mean = FALSE)) ## EGARCH model
#specg = ugarchspec(variance.model = list(model="sGARCH", garchOrder = c(5,0)), mean.model = list(armaOrder = c(0,0), include.mean = FALSE))  ## ARCH model (by setting appropriate GARCH coefs to zero)

for (i in train:mysize){
  
  y22 = tf[1:i-1,1];
  fit = ugarchfit(data = y22, spec = specg)
  forc = ugarchforecast(fit, n.ahead=1)
  pr = as.numeric(forc@forecast["sigmaFor"])
  #prm=as.numeric(forc@forecast["seriesFor"])
  
  ll= ll + dnorm(tf[i,1], mean =0, sd = pr, log = TRUE)
  
}

ll
