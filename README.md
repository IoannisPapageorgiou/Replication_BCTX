# Replication package for "The Bayesian Context Trees State Space Model for time series modelling and forecasting"

Ioannis Papageorgiou and Ioannis Kontoyiannis

## Overview and contents

The code in this repository generates the forecasting results in the paper "The Bayesian Context Trees State Space Model for time series modelling and forecasting", and more specifically, those presented in Tables 1-3 and Figure 5.

The main contents of this repository are organised in the following folders:

* table1
* table2
* table3
* fig5

where one of these folders contains: 

1) The data used in generating corresponding table/figure
2) The code used to generate the BCT-X forecasting results
3) The code used to generate the forecasting results of alternative methods

## Instructions and computational requirements

Since an important advantage of the methodology proposed in this paper is its low computational (time and space) requirements, all experiments were carried out on the CPU of a common laptop.

The proposed BCT-X methodology is implemented in C++ code for further computational efficiency. A different ".cpp" file is included in each folder which generates the BCT-X forecasting results of the correspnding figure/table. The data read by the C++ code (at the start of the main function) should be at the same directory with the ".cpp" file.

Altrnative methods are implented in R and Python. A complete description of these alternatives approaches and their corresponding dependencies are given in each subfolder, and further details can be found in the main paper and the Supplementary Material (Section E). Important resources include the R package 'forecast', which implements common and widely-used forecasting approaches, the python libraby 'GluonTS', which implements a number of machine learning approaches, and the R package 'rugarch', which implements volatility models.

## Data availability and provenance

A complete list of all the datasets considered in this paper is given in Section C of the Supplemental Material of the paper. This includes both a description of each dataset as well as their original sources. In the above directories, the datasets are included in the corresponding figure/table in which they are (first) used.

Apart from the simulated datasets that were created for the purpose of this paper and are now publically releashed in this repository, the real-world applications considered in this paper are common, publicly available data. The vast majority of these consist of standard daily stock prices (either of individual stocks or stock market indices), which are widely available online, and are in some cases tranformed (taking logs, differences, etc) to create the datasets used in the paper; see Section C of the Supplemental Material of the paper for more details. The only datasets of different kind are the Unemployment Rate and GNP of the US, which are available for example from the US Bureau of Labor Statistics (BLS) and the Federal Reserve Bank of St. Louis (FRED), resepctively; see links the Supplemental material. 

## References

* I. Papageorgiou and I. Kontoyiannis. The Bayesian Context Trees State Space Model for time series
modelling and forecasting. Submitted, International Journal of Forecasting, 2025.

* R.J. Hyndman and Y. Khandakar. Automatic time series forecasting: The forecast package for R.
J. Stat. Softw., 26(3):1–22, 2008.

* A. Alexandrov, et al. GluonTS:
Probabilistic and neural time series modeling in Python. J. of Mach. Learn. Res., 21(116):1–6, 2020.

* A. Ghalanos. rugarch: Univariate GARCH models. R package version 1.4, 2022. Available
at CRAN.R-project.org/package=rugarch.




