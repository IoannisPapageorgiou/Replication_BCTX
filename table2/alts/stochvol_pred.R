library(stochvol)

#tf is data
tf=read.delim("../data/ftse.txt", header = FALSE) #can similarly replace this with other datasets

mysize=nrow(tf);
train=nrow(tf)-130+1;

ll=0


for (i in train:mysize){
  
  
#print(i)
y22 = tf[1:i-1,1];


#drawsar <- svsample(y22, draws = 1000, burnin = 100,
                    #designmatrix = "ar1", priormu = c(0, 1.0), priorphi = c(1.0,1.0),
                    #priorsigma = 1.0)

drawsar <- svsample(y22, draws = 1000, burnin = 100, priormu = c(0, 1.0), priorsigma = 0.1, priorphi = c(1.0,1.0))


forear <- predict(drawsar, 1)


prm = mean(as.numeric(unlist(forear[["y"]])))
prv = mean(as.numeric(unlist(forear[["vol"]])))


ll= ll + dnorm(tf[i,1], mean = prm, sd = prv, log = TRUE)

}

ll
