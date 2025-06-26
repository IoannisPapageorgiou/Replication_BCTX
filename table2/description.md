In this folder the results of Table 2 are replicated.

The folder "data" includes the datasets used in this experiment, i.e., the (appropriately transformed) daily values of important stock market indices: FTSE 100, CAC 40, DAX and S&P 500.

The folder "bct_ar" contains the C++ code for the BCT-AR model in the above foecasting experiments. For simplicity and ease of preproduction, a different ".cpp" file is given for each real-world dataset, in which all the BCT-AR hyperparameters are already tuned appropriately.

The folder "alts" contains implementations of the alternative methods in R and Python. More specifically, the methods ARIMA, ETS and NNAR are implemented in 'forecast', and the methods DeepAR and N-BEATS are implemented in Python using the library 'GluonTS'. The MSA model is implemented using the R package 'MSwM', the SETAR model is implemented in the R package 'TSA', and finally the MAR model is implemeted using the R package 'mixAR'.
