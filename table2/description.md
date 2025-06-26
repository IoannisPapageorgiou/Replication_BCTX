In this folder the results of Table 2 are replicated.

The folder "data" includes the datasets used in this experiment, i.e., the (appropriately transformed) daily values of important stock market indices: FTSE 100, CAC 40, DAX and S&P 500.

The folder "bct_arch" contains the C++ code for the BCT-ARCH model in the above forecasting experiment. The dataset is read in the main function: the first dataset 'ftse.txt' is read here, but the other datasets can be loaded similarly. 

The folder "alts" contains implementations of the alternative methods in R. More specifically, the methods ARCH, GARCH, GJR, EGARCH are implemented using the R package 'rugarch', while the MSGARCH model is implemented using the package 'MSGARCH', and the SV model is implemented using the package 'stochvol'.
