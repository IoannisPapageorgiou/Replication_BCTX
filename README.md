# Replication package for "The Bayesian Context Trees State Space Model for time series modelling and forecasting"

Ioannis Papageorgiou and Ioannis Kontoyiannis

## Overview and contents

The code in this repository generates the forecasting results in the paper "The Bayesian Context Trees State Space Model for time series modelling and forecasting", and more specifically, those presented in Tables 1-3 and Figure 5.

The main contents of this repository are organised in the following folders:

* table1
* table2
* table3
* fig5

Each one of these folders contains: 

1) The data used in generating corresponding table/figure
2) The code used to generate the BCT-X forecasting results
3) The code used to generate the forecasting results of alternative methods

## Instructions and computational requirements

Since an important advantage of the methodology proposed in this paper is its low computational (time and space) requirements, all experiments were carried out on the CPU of a common laptop.

The proposed BCT-X methodology is implemented in C++ code for further computational efficiency. A different .cpp file is included in each folder which generates the BCT-X forecasting results of the correspnding figure/table.

Altrnative methods are implented in R and Python. A complete description of these alternatives approaches and their corresponding dependencies are given in each subfolder, and further details can be found in the main paper and the Supplementary Material (Section E). Important resources include the R package 'forecast', which implements common and widely-used forecasting approaches, the python libraby 'GluonTS', which implements a number of machine learning approaches, and the R package 'rugarch', which implements volatility models.
