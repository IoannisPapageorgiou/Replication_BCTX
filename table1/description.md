In this folder the results of Table 1 are replicated.

The folder "data" includes the datasets used in this experiment: the simulated datasets sim1, sim2, sim3, and the real-world datasets unemp, gnp, ibm. All real-world datasets contain the transformed (differenced, etc.) data, so they just need to be loaded in their exact form.

The folder "bct_code" contains the C++ code for the BCT-AR model in the above foecasting experiments. For simplicity and ease of preproduction, a different ".cpp" file is given for each dataset, in which all the BCT-AR hyperparameters are already tuned appropriately.  

The folder "alts" contains implementations of the alternative methods in R and Python. These files automatically load the data from the folder "table1/data" and compute the MSE results from the corresponding method. Specifically:

* The file 'alts/forecast_pred.R' implements the methods ARIMA, ETS, and NNAR, using the R package 'forecast'.

* The file 'alts/gluonts_pred.py' is Python code which implements the methods DeepAR and N-BEATS, using the library 'GluonTS'.

* The file 'alts/msa_pred.R' implements the MSA model using the R package 'MSwM'.

* The file 'alts/mar_pred.R' implements the MAR model using the R package 'mixAR'.

* The file 'alts/tar_pred.R' implements the SETAR model using the R package 'TSA'.
