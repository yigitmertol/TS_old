import pandas as pd
import matplotlib
import preprocessing_SNP
from matplotlib import pyplot as plt

prcs = preprocessing_SNP.processor()
prcs.init(0)

markets = dict()
for market in ["AAP"]:
    prcs.load_SNP(market)
    # prcs.make_feats()
    markets[market] = prcs.raw_data
for market in markets:
    series = markets[market][0:10]
    # x = [int(t) for t in range(len(series))]
    # y = [float(c) for c in series['close']]
    plt.plot(series['date'], series['close'])
    plt.savefig("figure.png")

