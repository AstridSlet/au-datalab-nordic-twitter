import os
import json
import pandas as pd 
import re 
import demoji
import numpy as np
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# define labels and their numeric equvalents
labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

# load classifier and tokenizer 
classifier = AutoModelForSequenceClassification.from_pretrained("datalab/xlm-roberta-base_lr2e5_e4_esc", use_auth_token="xxx",problem_type="multi_label_classification",num_labels=len(labels),id2label=id2label,label2id=label2id)
tokenizer = AutoTokenizer.from_pretrained("datalab/xlm-roberta-base_lr2e5_e4_esc", use_auth_token="xxx", return_tensors="pt")

# function for getting probabilities
def predict_proba(text):
        encoding = tokenizer(text, return_tensors="pt")
        encoding = {k: v.to(classifier.device) for k,v in encoding.items()}
        outputs = classifier(**encoding)
        logits = outputs.logits
        # apply sigmoid + threshold
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        probs = probs.squeeze().tolist()
        return probs


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

country = 'no'
dirName = '/work/76568/preprocessed/'+country
listOfFiles = getListOfFiles(dirName)

print(f'[INFO] number of files in total: {len(listOfFiles)}')   

# loop over files seperately
for file_name in listOfFiles:
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

    # preprocess tweets
    tweets = df.text.values
    tweets = [text for text in tweets] ## Switch to normal way to turn this into a list 
    print('list of tweets created for file:' + file_name)

    # remove mentions and urls
    clean_tweets = [re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", str(tweet)) for tweet in tweets]
    # removen whitespaces created
    clean_tweets = [re.sub(r"^\s+|\s+$", "", str(tweet)) for tweet in clean_tweets]

    # decode emojis and save to list of dics
    tweets_demoji= [demoji.findall(tweet) for tweet in clean_tweets]

    # Select values (ignore keys)  and save to list
    demoji_list = []
    for dic in tweets_demoji:
        if bool(dic) == False:
            demoji_list.append('')
        elif bool(dic) == True:
            tmp = list(dic.values())[0]
            demoji_list.append(tmp)

    # combine clean tweets and decoded emojis into new list 
    clean_tweets_demoji = []
    for i in range(len(clean_tweets)):
        clean_tweets_demoji.append(clean_tweets[i] + ' ' + demoji_list[i])

    
    ### add column to existing dataframe with clean test used for emotion detection and the main emotion extracted
    df['clean_text_demoji'] = clean_tweets_demoji
    print('[INFO] Finished cleaning text!')

    # predict emotion probabilities
    eprob = [predict_proba(tweet) for tweet in clean_tweets_demoji]

    # Create list for each emotion probability 
    anger=[item[0] for item in eprob]
    anticipation=[item[1] for item in eprob]
    disgust=[item[2] for item in eprob]
    fear=[item[3] for item in eprob]
    joy=[item[4] for item in eprob]
    love=[item[5] for item in eprob]
    optimism=[item[6] for item in eprob]
    pessimism=[item[7] for item in eprob]
    sadness =[item[8] for item in eprob]
    suprise =[item[9] for item in eprob]
    trust =[item[10] for item in eprob]

# Add lists to existing dataframe 
    df['joy'] = joy
    df['trust'] = trust
    df['anticipation'] = anticipation
    df['surprise'] = suprise
    df['anger'] = anger
    df['disgust'] = disgust
    df['sadness'] = sadness
    df['fear'] = fear
    # the three extra emotions
    df['optimism'] = optimism
    df['pessimism'] = pessimism
    df['love'] = love

# save to new file
    print('[INFO] ready to save file:' + file_name)
    df.to_json(file_name, orient="records",lines=True)

    with open("completed_"+country+"_ED.txt", "a") as external_file:
        print(file_name, file=external_file)
        external_file.close()
    
    print('[INFO] Finished with this file')
    end = time.time()
    print(f'[INFO] time: {end - start}')

print('[INFO] loop finished')
