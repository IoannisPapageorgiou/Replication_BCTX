In this folder the results of Table 1 are replicated.

The datasets are in the folder "data" and include: the simulated datasets sim1, sim2, sim3, and the real-world datasets unemp, gnp, ibm. All real-world datasets in this folder contain the transformed (differenced, etc) data, so they just need to be loaded in their exact form.

The folder "bctx" contains the C++ code for the BCT-AR model in the above foecasting experiments. For simplicity and ease of preproduction, a different ".cpp" file is given for each dataset, in which all the BCT-AR hyperparameters are already tuned appropriately. 

The folder "bench" contains implementations of the alternative methods in R and Python for the same datasets to be loaded. More specifically, hte methods ARIMA, ETS and NNAR are implemented in 'forecast', the methods DeepAR and N-BEATS are implemented in Python using the library 'GLuonTS', the method MSA is implemented using the R package 'MSwM', the method SETAR is implemented in the R package 'TSA', and finally MAR is implemeted using the R package 'mixAR'. 
