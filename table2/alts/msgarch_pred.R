library(MSGARCH)

tf=read.delim("ftse.txt", header = FALSE) # read data
tfnum=as.numeric(unlist(tf))


mysize=nrow(tf);
train=nrow(tf)-130+1;


ll=0



spec_ms <- CreateSpec(variance.spec = list(model = c("sGARCH", "sGARCH")),distribution.spec = list(distribution = c("norm", "norm")))
#spec_ms <- CreateSpec(variance.spec = list(model = c("sGARCH", "sGARCH", "sGARCH")),distribution.spec = list(distribution = c("norm", "norm", "norm")))

for (i in train:mysize){
  
  
  #print(i)
  y22 = tfnum[1:i-1];
  
  
  fitms = FitML(spec = spec_ms, data =y22)
  
  
  prv= as.numeric(unlist(predict(fitms, nahead=1)))
  
  
  ll= ll + dnorm(tf[i,1], mean = 0, sd = prv, log = TRUE)
  
}


ll