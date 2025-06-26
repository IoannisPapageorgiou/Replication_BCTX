
tf=read.delim("data.txt")

n=0;
mysize=nrow(tf);
train=nrow(tf)-130+1;


ll=0
mse =0
for (i in train:mysize){
  
  
print(i)
y22 = tf[1:i-1,1];


## Obtain 7000 draws
#drawsar <- svsample(y22, draws = 1000, burnin = 100,
                    #designmatrix = "ar1", priormu = c(0, 1.0), priorphi = c(1.0,1.0),
                    #priorsigma = 1.0)

drawsar <- svsample(y22, draws = 1000, burnin = 100, priormu = c(0, 1.0), priorsigma = 0.1, priorphi = c(1.0,1.0))

## Predict 7 days ahead (using AR(1) mean for the returns)
forear <- predict(drawsar, 1)

#fp=forear[["y"]]
#pr_vol=forear[["vol"]]


prm = mean(as.numeric(unlist(forear[["y"]])))
prv = mean(as.numeric(unlist(forear[["vol"]])))



#er = pr - y2[i+2]
#mse = mse + er*er


ll= ll + dnorm(tf[i,1], mean = prm, sd = prv, log = TRUE)

}



ll