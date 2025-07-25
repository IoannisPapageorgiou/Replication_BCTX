# Replication package for "The Bayesian Context Trees State Space Model for time series modelling and forecasting"

Ioannis Papageorgiou and Ioannis Kontoyiannis

## Overview, contents and instructions

The code in this repository generates the forecasting results in the paper "The Bayesian Context Trees State Space Model for time series modelling and forecasting", and more specifically, those presented in Tables 1-3 and Figure 5.

The main contents of this repository are organised in the following folders:

* table1
* table2
* table3
* fig5

Each of these folders contains everything that is required to reproduce the results of the corresponding table/figure, which for  convenience are organised as follows:

1) A "readme" file that gives specific instructions to reproduce the results of the corresponding experiment.
1) A subfolder named "data" that containts all datasets used in generating the corresponding table/figure.
2) A subfolder named "bct_code" that contains the code used to generate the BCT-X forecasting results.
3) A subfolder named "alts" that contains the code used to generate the forecasting results of all the alternative methods.

## Computational requirements and dependencies

Since an important advantage of the methodology proposed in this paper is its low computational requirements (time and space), all experiments were carried out on the CPU of a common laptop.

The proposed BCT-X methodology is implemented in C++ for further computational efficiency. A different ".cpp" file is included in each folder which generates the BCT-X forecasting results of the correspnding figure/table; the data read by the C++ code should be at the same directory with the ".cpp" file. This C++ implementation has been built using the linear algebra library 'Eigen' (version 3.3.9, available at: https://eigen.tuxfamily.org/index.php?title=Main_Page), which needs to be included as an additional directory. 

Altrnative methods are implented in R and Python. A complete description of these alternatives approaches and their corresponding dependencies are given in each subfolder, and further details can be found in the main paper and the Supplementary Material (Section E). Important resources include the R package 'forecast', which implements common and widely-used forecasting approaches, the python library 'GluonTS', which implements a number of machine learning approaches, and the R package 'rugarch', which implements volatility models.

## Data availability and provenance

A complete list of all the datasets examined in this paper is given in Section C of the Supplementary Material of the paper. This includes both a description of each dataset as well as their original sources. In the above directories, the datasets are included in the corresponding figure/table folder in which they appear.

Apart from the simulated datasets that were created for the purpose of this paper and are now publicly releashed in this repository, the real-world applications considered in this paper are using common, publicly available data. The vast majority of these consist of standard daily stock prices (either of individual stocks or stock market indices), which are widely available online, and are in some cases tranformed (taking logs, differences, etc) to create the datasets that are finally used in the paper; see Section C of the Supplementary Material of the paper for more details. The only datasets of different kind are the Unemployment Rate and GNP of the US, which are available for example from the US Bureau of Labor Statistics (BLS) and the Federal Reserve Bank of St. Louis (FRED), resepctively; see links the Supplementary material. 

## References

* I. Papageorgiou and I. Kontoyiannis. The Bayesian Context Trees State Space Model for time series
modelling and forecasting. Submitted, International Journal of Forecasting, 2025.

* R.J. Hyndman and Y. Khandakar. Automatic time series forecasting: The forecast package for R.
J. Stat. Softw., 26(3):1–22, 2008.

* A. Alexandrov, et al. GluonTS:
Probabilistic and neural time series modeling in Python. J. of Mach. Learn. Res., 21(116):1–6, 2020.

* A. Ghalanos. rugarch: Univariate GARCH models. R package version 1.4, 2022. 




