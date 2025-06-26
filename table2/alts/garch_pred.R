library(rugarch)

#tf is data
tf=read.delim("ftse.txt", header = FALSE) # can similarly replace this with other datasets


mysize=nrow(tf);
train=nrow(tf)-130+1;


ll=0


specg = ugarchspec(variance.model = list(model="sGARCH")) ## GARCH model
#specg = ugarchspec(variance.model = list(model="gjrGARCH")) ## GJR model
#specg = ugarchspec(variance.model = list(model="eGARCH")) ## EGARCH model 
#specg = ugarchspec(variance.model = list(model="sGARCH", garchOrder = c(5,0))) ## ARCH model (by setting appropriate GARCH coefs to zero)




for (i in train:mysize){
  
  #print(i)
  
  y22 = tf[1:i-1,1];
  fit = ugarchfit(data = y22, spec = specg)
  forc = ugarchforecast(fit, n.ahead=1)
  pr = as.numeric(forc@forecast["sigmaFor"])
  prm=as.numeric(forc@forecast["seriesFor"])
  
  
  ll= ll + dnorm(tf[i,1], mean = prm, sd = pr, log = TRUE)
  
}

ll