from Processors import preprocessing_SNP
import os
import numpy as np
import pandas as pd

prcs = preprocessing_SNP.processor()
prcs.init()
for file in os.listdir('../sandp500/individual_stocks_5yr'):
    if file in[".DS_Store"] :
        continue
    market = file.split('_')[0]
    prcs.load_SNP(market)
    if prcs.Data_loaded.shape[0] < 1250:
        continue
    for col in prcs.Data_loaded.columns:
        if any(pd.isna(prcs.Data_loaded[col])):
            prcs.handle_nans()

    prcs.make_feats(True)

