library(MSwM)

mse = 0;
pred_step =1;


xn = read.csv("../data/sim1.txt", header = FALSE) ## read desired dataset here
l = length(xn[,1])
l2 = l/2

for (i in l2:l){
  
  #print(i)
  y22 = xn[1:i-1,1];
  model22 = lm(y22~1);
  fitted = msmFit(object =model22 ,  k=2 , sw = c(T,T,T,T) , p =2  ) # MSA model parameters 
  coef = fitted["Coef"]
  c = as.numeric(unlist(coef))
  
  matrixx =  fitted["transMat"];
  m= as.numeric(matrixx);
  
  f1 = fitted["Fit"]["smoProb"][i-2,1]; #smooth up to time 
  f2 = fitted["Fit"]["smoProb"][i-2,2];
  
  
  p1 = f1* m[1] + f2 *m[3];
  p2 = f1 * m[2] + f2 * m[4];
  
  
  
  prd = p1 * (c[1] + c[3] * y22[i-1] + c[5] * y22[i-2]) + p2 * (c[2] + c[4] * y22[i-1] + c[6] * y22[i-2] );
  
  er = prd-xn[i,1]; 
  mse = mse + er*er;
  
  
  
}

mse / l2
  
  
