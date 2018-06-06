import pandas as pd
import matplotlib
import preprocessing
from matplotlib import pyplot as plt

prcs = preprocessing.processor()
prcs.init(0)

markets = dict()
for market in ["AAL"]:
    prcs.load_SNP(market)
    # prcs.make_feats()
    markets[market] = prcs.all_data
for market in markets:
    series = markets[market]
    # x = [int(t) for t in range(len(series))]
    # y = [float(c) for c in series['close']]
    plt.plot(series['date'], series['close'])
