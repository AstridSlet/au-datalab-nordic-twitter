#!/usr/bin/env python
# coding: utf-8

import os
import json
import pandas as pd 
import re 
import numpy as np
import time

# function for reading all files
def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)  
    return allFiles

dirName = '/work/76568/preprocessed/'
country = 'no'

listOfFiles = getListOfFiles(dirName+country) 

# read files
tweets = []
for file in listOfFiles: 
    print('[INFO] reading file:' + file)
    for line in open(file, 'r', encoding='utf-8'):
        try:
            tweets.append(json.loads(line))
        except:
            pass
df=pd.DataFrame(tweets)
print(f'df created with n lines: {df.shape[0]}')

# locate retweets with mask 
rt_terms = ["RT ", "RT:", "RT :", "RT", "RT : "] # tweets that start with these
mask = list()
for l in df.text.astype(str).apply(lambda x : x.split()): 
    if any(w in rt_terms for w in l[:2]): # for each textline (list of str) check if any of first 3 tokens correspond to any of the 'rt' terms
        mask.append(True)
    else: mask.append(False)
retweets = df[mask].copy(True)
print(f'[INFO] n retweets: {retweets.shape}')
retweets.to_json(dirName+f'/'+country+'_split/'+country+'_retweets_new.ndjson', orient="records",lines=True)
 

# original tweets
opposite_mask = [not elem for elem in mask]
original = df[opposite_mask].copy(True)
print(f'[INFO] n original tweets: {original.shape}')
original.to_json(dirName+f'/'+country+'_split/'+country+'_original_new.ndjson', orient="records",lines=True)

# add to logfile 
with open(dirName+f'/'+country+'_split/'+country+'_stats_new.txt', "a") as external_file:
    print(f"Number of original tweets: {str(original.shape)}\n", file=external_file)
    print(f"Number of retweets: {str(retweets.shape)}\n", file=external_file)
    external_file.close()
    
print('[INFO] loop finished')
