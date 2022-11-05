#!/usr/bin/env python
# coding: utf-8

import os
import json
import pandas as pd 
import re 
import numpy as np
import time

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


# get list of files 
country = 'sv'
dirName = '/work/76568/preprocessed/' + country
listOfFiles = getListOfFiles(dirName) 

# list of parsed files
with open('completed_'+country+'_dyads.txt') as f:
    parsed = f.read().splitlines()
    f.close()
not_completed_files = [f for f in listOfFiles if f not in parsed]
print(f"{len(not_completed_files)} out of {len(listOfFiles)} files are not yet completed")

# emotions file
df_em = pd.read_excel('/work/AstridSlettenRybner#9771/plutchiks_emotions.xlsx', engine='openpyxl')

# emotion columns
cols = ['joy', 'trust', 'anticipation', 'surprise', 'anger', 'disgust', 'sadness', 'fear', 'optimism', 'pessimism', 'love']

for file_name in not_completed_files:
    start = time.time()
    tweets = []
    for line in open(file_name, 'r'):
        try:
            tweets.append(json.loads(line))
        except:
            pass
    df=pd.DataFrame(tweets)
    print('[INFO] df created for file:' + file_name)
    print(f'with n lines: {df.shape[0]}')
    
    # find top 2 emotions for each tweet (of the emotional tweets)
    top1 = df.loc[:,cols].apply(lambda row: row.nlargest(2).index[-0],axis=1).rename("top1")
    top1_score = df.loc[:,cols].apply(lambda row: row.nlargest(2).values[-0],axis=1).rename("top1_score")
    top2 = df.loc[:,cols].apply(lambda row: row.nlargest(2).index[-1],axis=1).rename("top2")
    top2_score = df.loc[:,cols].apply(lambda row: row.nlargest(2).values[-1],axis=1).rename("top2_score")
    

    # add to df
    df_combined = pd.concat([df, top1, top1_score, top2, top2_score], axis = 1)
    
    # files for storing detected dyads
    idx = []
    dyad = []
    category = []

    # detecting dyads
    for index1, row1 in df_combined.iterrows():
        top_emotions = [df_combined['top1'][index1], df_combined['top2'][index1]]
        # loop over emotion combination pairs
        for index2, row2 in df_em.iterrows():
            emotion_pair = [str(df_em['em1'][index2]), str(df_em['em2'][index2])]
            # check if both elements of top emotions are in emotion pair
            if all(any(sub in string for string in list(top_emotions)) for sub in emotion_pair):
                idx.append(index1)
                dyad.append(df_em['dyad'][index2])
                category.append(df_em['category'][index2])
            else:
                 pass
    
    # save detections
    detections = {'id': idx, 'dyad': dyad, 'category':category}
    detections_df = pd.DataFrame(data=detections)
    detections_df = detections_df.set_index('id')
    
    # add to df with all emotional tweets
    allDetections = pd.concat([df_combined, detections_df], axis=1).fillna(0)
    
    # add newly created columns to original df (with emotional and non emotional tweets)
    colsNew = ['top1','top1_score', 'top2', 'top2_score', 'dyad', 'category']
    allDetections = allDetections.loc[:, colsNew]
    final = pd.concat([df, allDetections], axis=1).fillna(0)
    
    # save to new file
    print('[INFO] ready to save file:' + file_name)
    final.to_json(file_name, orient="records",lines=True)
    end = time.time()
    print(f'[INFO] time: {end - start}')
    
    # add to logfile 
    with open("completed_"+country+"_dyads.txt", "a") as external_file:
        print(file_name, file=external_file)
        external_file.close()

