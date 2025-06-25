library(TSA)

mse = 0;
pred_step =1;


xn = read.csv("sim1.txt", header = FALSE) ## read desired dataset here
l = length(xn[,1])
l2 = l/2


for (n in l:l2){
  
  
  #update model every time-step
  temp = xn[1:(n-1),1]; 
  dts = ts(temp);
  
  
  my_tar = tar(dts, p1 = 2, p2 = 2, d= 1,  method = 'CLS'); ## SETAR model parameters

  
  pr = predict(my_tar,1,1000);
  y = unname(unlist(pr["fit"]));
  er = y[pred_step]-xn[n,1];
  
  ###################################################################
  
  mse = mse + er^2;
  
}

mse/l2
