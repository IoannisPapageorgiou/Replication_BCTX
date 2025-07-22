library(forecast)

mse = 0;

xn = read.csv("../data/sim1.txt", header = FALSE) ## read desired dataset; here sim1 is used, but can similarly use with the other datasets.
n = length(xn[,1])
n2 = n/2

for (i in n2:n){
  
  #update model every time-step
  temp = xn[1:(i-1),1];
  dts = ts(temp);
  ar = auto.arima(dts);  ## this is for ARIMA model
  #ar = ets(dts); ## this is for ETS model
  #ar = nnetar(dts,p=2); ## this is for NNAR model

  f = forecast(ar);
  p  = unname(unlist(f["mean"]));
  er = p[1]-xn[i,1];

  mse = mse + er^2
}

mse / n2




