import pandas as pd
from Processors import preprocessing_SNP
from matplotlib import pyplot as plt
import os
import numpy as np


prcs = preprocessing_SNP.processor()
#
markets = dict()
# prcs.load_SNP("ABBV")
# markets["ABBV"] = prcs.Data_loaded
for file_name in os.listdir("../Data/SnP_processed")[9:12]:
    market = file_name.split('_')[2]
    prcs.load_SNP_processed(market)
    markets[market] = prcs.Data_loaded
for market in markets:
    # if market not in ["ABBV"]:
    #     continue
    series = markets[market][0:1000]
    # series = markets[market][prcs.Ts['dev'][0]:prcs.Ts['dev'][-1]]
    # x = [int(t) for t in range(len(series))]
    # y = [float(c) for c in series['close']]
    fig, ax1 = plt.subplots()

    # Plot bars
    ax1.bar(series[series.columns[0]], series['return'], alpha=1)
    ax1.set_xlabel('$t$')

    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('bar', color='b')
    [tl.set_color('b') for tl in ax1.get_yticklabels()]

    # Set up ax2 to be the second y axis with x shared
    ax2 = ax1.twinx()
    # Plot a line
    ax2.plot(series[series.columns[0]], series['close'], 'ro--', linewidth=0.6, markersize=0.6)
    plt.ylim(0.5*min(series['close']), 1.2*max(series['close']))

    import matplotlib.dates as mdates

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gcf().autofmt_xdate()
    # Make the y-axis label and tick labels match the line color.
    ax2.set_ylabel('close price', color='r')
    [tl.set_color('r') for tl in ax2.get_yticklabels()]

    ax2.grid(color='r', linestyle='-', linewidth=0.3)

    plt.show()

    # plt.plot(series['t'], series['close'])
    # plt.bar(series['t'], series['volume'])
    # plt.
    # plt.savefig("./SnP_figures/" + str(market) + "_close.png")

