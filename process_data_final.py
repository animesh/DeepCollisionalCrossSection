#!/usr/bin/env python
# coding: utf-8
#https://colab.research.google.com/github/animesh/DeepCollisionalCrossSection/blob/master/process_data_final.ipynb#scrollTo=kMrHXX61ISRz
import sys
data_path = sys.argv[1]
outpath = data_path+'_proc_'
import pandas as pd
df=pd.read_table(data_path)
print(df.head())
print(df["Score"].count())
print(df["Missed cleavages"].count())
print(df.shape)
df = df.rename(index=str, columns={"Modified sequence": "Modified_sequence"})
df['Modified_sequence'] = df['Modified_sequence'].str.replace('_','')
print(df['Modified_sequence'].count())

import numpy as np
low = np.percentile(np.min(np.log2(df['Intensity'])), 10)
high = np.percentile(np.max(np.log2(df['Intensity'])), 90)
print(low, high)

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
data=df
dat = data['Modified_sequence']
dat = [list(d) for d in dat]
#process data into one hot encoding
flat_list = ['_'] + [item for sublist in dat for item in sublist]
# define example
values = np.array(flat_list)
label_encoder = LabelEncoder()
label_encoder.fit(values)
print(values,label_encoder.classes_, len(label_encoder.classes_),label_encoder.transform([['_']]))

import re, os, csv
import pickle
with open('enc.pickle', 'wb') as handle:
    pickle.dump(label_encoder, handle)
import csv
with open('enc_list.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(list(label_encoder.classes_))

df.groupby(['Modified_sequence', 'Charge'], group_keys=False).count()#apply(lambda x: x.loc[x.Intensity.idxmax()])
dfMaxS=df.groupby(['Modified_sequence', 'Charge'], group_keys=False).apply(lambda x: x.loc[x.Score.idxmax()])
#dfMaxS["Score"].hist()
dfMaxS['label']=dfMaxS['Score'].values.tolist()

def split(data, name, s, label_encoder_path='enc.pickle', ids=None, calc_minval=True):
    np.random.seed(s)
    with open(label_encoder_path, 'rb') as handle:
        label_encoder = pickle.load(handle)
    data['encseq'] = data['Modified_sequence'].apply(lambda x: label_encoder.transform(list(x)))
    if calc_minval:
        data['minval'] = np.min(data['label'])
        data['maxval'] = np.max(data['label'])
    else:
        data['minval']=-2
        data['maxval']=325
    data['task'] = 0
    print('Name: ', name, 'Seed: ', s, 'Len test: ', len(data[data['test']]),'Len set test: ', len(set(data[data['test']])),'Len not test: ', len(data[~data['test']]),'Len set not test: ', len(set(data[~data['test']])))
    data[~data['test']].to_pickle(name + str(s) + '_train.pkl')
    data[data['test']].to_pickle(name +str(s) + '_test.pkl')
    return data

import random
dfMaxS['test']=np.random.choice([True, False], dfMaxS.shape[0])#bool(random.getrandbits(1))
dd=split(dfMaxS, outpath, 2)

trainseqs = dd[~dd['test']]['Modified_sequence'].values.tolist()
ddd = dd[dd['test']]
ddd[ddd['Modified_sequence'].isin(trainseqs)].shape
print(len(ddd))
