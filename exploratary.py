import pandas as pd
import matplotlib
import preprocessing_SNP
from matplotlib import pyplot as plt
import os

prcs = preprocessing_SNP.processor()
prcs.init(batch_size=10, _input_window_l=10, _horizon=1)


markets = dict()
for file_name in os.listdir("sandp500/individual_stocks_5yr")[1:10]:
    market = file_name.split('_')[0]
    prcs.load_SNP_processed(market)
    markets[market] = prcs.with_feats
for market in markets:

    series = markets[market]
    # x = [int(t) for t in range(len(series))]
    # y = [float(c) for c in series['close']]
    plt.plot(series['t'], series['close'])
    # plt.
    plt.savefig("./SnP_figures/" + str(market) + "_close.png")

