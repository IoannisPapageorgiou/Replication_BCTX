# This first part loads the results from Table 2, i.e., for FTSE 100, CAC 40, DAX and S&P 500

d2 = c(-161.9, -112.5, -111.7, -78.73, -159.7, -111, -106.4, -83.89, -159, -112.4, -107.5, -84.58, -159.7, -109.2, -106.1, -80.95)

x3 = matrix(d2, 4,4)

colnames(x3) <- c("BCT-ARCH",  "GJR", "EGARCH", "MSGARCH" )


# This part loads the remaining results from Table 3, for a number of important stock market indices.
# Note that a minus sign is added to all results, since we want the best method to appear as first in the tests.


x3 = rbind(x3, c(-104.4, -99.27, -102.5, -102.6)) # euro Stoxx 50

x3 = rbind(x3, c(-132.3, -128.4, -130.9, -130.3)) # Stoxx europe 600

x3 = rbind(x3, c(-128.5, -125.1, -123.8, -127.3)) # edow 

x3 = rbind(x3, c(-125.7, -122.8 , -119.0, -122.1)) # austria

x3 = rbind(x3, c(-28.80, -18.96,-21.22,  -21.02)) #  denmark

x3 = rbind(x3, c(-111.4, -107.3, -110.4, -110.2)) # aex (dutsch)

x3 = rbind(x3, c(-97.28, -94.58, -96.06, -94.48)) # italian

x3 = rbind(x3, c(-129.0, -125.5, -125.3, -122.4)) # ibex (spain)

x3 = rbind(x3, c(-134.8, -133.2, -132.9, -131.8)) # sweden

x3 = rbind(x3, c(-162.4 , -161.0 , -161.5, -161.0)) # bel 20

x3 = rbind(x3, c(-156.7, -158.7, -155.9, -158.9)) # swiss

x3 = rbind(x3, c(-150.1, -148.7, -149.2, -151.6 )) # portugal 

x3 = rbind(x3, c(-141.4, -146.2, -144.5, -145.6 )) # finland

x3 = rbind(x3, c(-135.9, -139.3, -140.3, -140.9 )) # norway

x3 = rbind(x3, c(-137.0, -137.1, -137.5, -125.9)) # greece

x3 = rbind(x3, c(-123.0 , -118.3, -117.3, -119.8)) # ireland

x3 = rbind(x3, c(-162.5, -160.3 , - 159.8 , -156.1 )) # ftse all

library(tsutils) ## this part implements the post-hoc Nemenyi tests using the R package 'tsutils'

nemenyi(x3,conf.level=0.9,plottype="mcb") ## this creates the left part of Figure 5, i.e., the MCB plot
#nemenyi(x3,conf.level=0.9,plottype="matrix") ## this creates the right part of Figure 5, i.e., the matrix plot
