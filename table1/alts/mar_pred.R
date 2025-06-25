library(mixAR)

mse=0;

pred_step =1;

xn = read.csv("sim1.txt", header = FALSE) ## read desired dataset here
l = length(xn[,1])
l2 = l/2


for (n in l:l2){
  
  #print(n)
  #update model every time-step
  temp = xn[1:(n-1),1]; 
  dts = ts(temp);
  
  
  mar = fit_mixAR(temp,model = c(2,2));
  pr=mar$model;
  
  past = c(sim_data[n-2], sim_data[n-1,1]);
  prf=multiStep_dist(pr,  maxh = 1, N = 1000 , xcond = past);
  y1 = prf(1, "location");

  er = y1-sim_data[n,1];

  mse = mse + er^2;

}

mse/l2
