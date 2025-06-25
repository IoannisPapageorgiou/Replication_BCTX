#matplotlib inline
import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.util import to_pandas

from gluonts.evaluation import make_evaluation_predictions
from gluonts.mx import Trainer

from gluonts.model.n_beats import NBEATSEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.gp_forecaster import GaussianProcessEstimator


# input here is the US GNP dataset read from 'gnp.txt'
# can similarly use with the other simulated and real-world datasets.

f = open("gnp_logdif.txt")
lines = f.readlines()
for i in list(range(0,291)):
    lines[i] = lines[i].strip()
    lines[i] = float(lines[i])

mse= 0


for i in list(range(146,291)): # size of training set is 50%

    temp_us_dif = np.array([lines[0:i+1]])
    
    
    prediction_length = 1 # 1-step ahead forecasts; can similarly do multi-step
    freq = "Q"
    start = pd.Timestamp("01-01-2019", freq=freq)  # can be different for each time series



    from gluonts.dataset.common import ListDataset

    # train dataset: cut the last window of length "prediction_length", add "target" and "start" fields
    train_ds = ListDataset(
        [{'target': x, 'start': start} for x in temp_us_dif[:, :-prediction_length]],
        freq=freq
    )
    # test dataset: use the whole dataset, add "target" and "start" fields
    test_ds = ListDataset(
        [{'target': x, 'start': start} for x in temp_us_dif],
        freq=freq
    )


    # For deepAR model
    estimator = DeepAREstimator(
         freq = "Q",
         prediction_length=1, # 1-step ahead forecasts; can similarly do multi-step
         context_length=1,    # tried between 1 and 5
         trainer=Trainer(
           ctx="cpu",
            epochs=5,
            learning_rate=1e-3,
            num_batches_per_epoch=50
            )
    )


    # For N-BEATS model
    #estimator = NBEATSEstimator(
         #freq = "Q",
         #prediction_length=1, # 1-step ahead forecasts; can similarly do multi-step
         #context_length=1,    # tried between 1 and 5
         #trainer=Trainer(
           # ctx="cpu",
           # epochs=3,
          # learning_rate=1e-3,
          #  num_batches_per_epoch=20
        #    )
    #)



    predictor = estimator.train(train_ds)




    forecast_it, ts_it = make_evaluation_predictions(
     dataset=test_ds,  # test dataset
     predictor=predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)


    # first entry of the time series list
    ts_entry = tss[0]

    # first entry of the forecast list
    forecast_entry = forecasts[0]

    #print(i)
    #print(f"Mean of the future window:\n {forecast_entry.mean}")
    #print(f"0.5-quantile (median) of the future window:\n {forecast_entry.quantile(0.5)}")

    er = forecast_entry.mean-lines[i]
    #er2 = forecast_entry.quantile(0.5)-lines[i] #using median for forecast


    #print(er*er)

    mse = mse + er*er

    

print(mse/146.0)



