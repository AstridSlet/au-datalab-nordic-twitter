import os
import json
import pandas as pd 
import re 
import numpy as np

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

# define function that checks if value is above or equal to 0.5
filterFunc = lambda x : 1 if x >= 0.5 else 0



dirName = '/work/76568/preprocessed/no'
outfile = '/work/76568/preprocessed/summarised/no_summarisedProportion.csv'

# retreive filepaths
listOfFiles = getListOfFiles(dirName) 
print(f'[INFO] loaded list of {len(listOfFiles)} files with following paths:')
print(listOfFiles)

# for storing all tweets
tweets = []

# loop over paths 
for file_name in listOfFiles:
    print("[INFO] reading file: " + file_name)
    for line in open(file_name, 'r'):
        try:
            tweets.append(json.loads(line))
        except:
            pass

# create df with all tweets
df=pd.DataFrame(tweets)
print(f'[INFO] combined df created for all files with n lines: {df.shape[0]}')

# fix date column
df['date'] = pd.to_datetime(df['created_at']).dt.date
    
# cols we want to transform to 0 and 1's 
colsOfInterest = ['joy', 'trust', 'anticipation', 'surprise', 'anger', 'disgust', 'sadness', 'fear', 'optimism', 'pessimism', 'love']

# apply the checking function to all the emotion columns
transformed = pd.DataFrame(df[colsOfInterest].applymap(lambda x: filterFunc(x)))

# concat with original df
final = pd.concat([pd.DataFrame(df.drop(colsOfInterest, axis=1)), transformed], axis="columns")

# group by date 
count = final[colsOfInterest].groupby([final['date']])

# calculate proportion of total n of tweets per day that contains each of the emotions 
summarised = pd.DataFrame((count.sum()/count.count())*100).reset_index()

# melt df
df_melted = pd.melt(summarised, 
                    id_vars=['date'],
                    value_vars=['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust','optimism', 'pessimism','love'],
                    var_name = 'primaryEmotion',
                    value_name = 'prop_percent')

# save summary to new file
print('[INFO] saving file' + outfile)
df_melted.to_csv(outfile)
