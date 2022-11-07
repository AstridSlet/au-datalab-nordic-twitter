from datasets import load_dataset
import pandas as pd 
import numpy as np
import os 

# load original datasets 
dataset_en = load_dataset("sem_eval_2018_task_1", "subtask5.english")
dataset_sp = load_dataset("sem_eval_2018_task_1", "subtask5.spanish")

# define directory to translated tweets
cwd = os.getcwd() # get current wd
directory = os.path.join(cwd, "my_SemEval") # add data folder 

train = [] # for storing datasets
test = []
val = []

# load datasets 
for f in os.listdir(directory):
    filepath = os.path.join(directory, f) # get path 
    name = filepath.split("/")[-1] # get name 
    print(f"[INFO] Loaded file: {name}")
    if not f.startswith('.'):
        if name.startswith('Train'): # if name starts with train 
            my_file = open(filepath, "r").read() # read file 
            train.append(my_file.split("\n")) # append to train list
        elif name.startswith('Test'):
            my_file = open(filepath, "r").read() 
            test.append(my_file.split("\n")) 
        elif name.startswith('Val'):
            my_file = open(filepath, "r").read() 
            val.append(my_file.split("\n")) 


# Input translated tweets and remove original collumn with english tweets into original english/spanish dataset

# english
train_en_FI=dataset_en['train'].add_column("Tweet_FI", train[0]).remove_columns("Tweet") 
test_en_FI=dataset_en['test'].add_column("Tweet_FI", test[1]).remove_columns("Tweet")
val_en_FI=dataset_en['validation'].add_column("Tweet_FI", val[0]).remove_columns("Tweet")

# spanish
train_sp_FI=dataset_sp['train'].add_column("Tweet_FI", train[1]).remove_columns("Tweet")
test_sp_FI=dataset_sp['test'].add_column("Tweet_FI", test[0]).remove_columns("Tweet")
val_sp_FI=dataset_sp['validation'].add_column("Tweet_FI", val[1]).remove_columns("Tweet")

### Recreate datasetDict structure 
from datasets import DatasetDict
from datasets import concatenate_datasets

# concatenate train, test and val sets
combined_train = concatenate_datasets([train_en_FI, train_sp_FI])
combined_test = concatenate_datasets([test_en_FI, test_sp_FI])
combined_val = concatenate_datasets([val_en_FI, val_sp_FI])

print(f"[INFO] Length of combined train: {len(combined_train)}")
print(f"[INFO] Length of combined test: {len(combined_test)}")
print(f"[INFO] Length of combined val: {len(combined_val)}")

# create dataset dict
dataset = DatasetDict ({'train':combined_train, 'test':combined_test, 'val': combined_val})

# Load the model and finetune it 

# Rename the tweets column 
dataset = dataset.rename_column("Tweet_FI", "Tweet")


## NB astrid only changed stuff in the above + changed the name of the model to be loaded ##

#create a list that contains the labels, as well as 2 dictionaries that map labels to integers and back.
labels = [label for label in dataset['train'].features.keys() if label not in ['ID', 'Tweet']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
print(labels)

# Preprocess the data 
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1") # Here I input my finnish bert 

def preprocess_data(examples):
  # take a batch of texts
  text = examples["Tweet"]
  # encode them
  encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
  # add labels
  labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
  # create numpy array of shape (batch_size, num_labels)
  labels_matrix = np.zeros((len(text), len(labels)))
  # fill numpy array
  for idx, label in enumerate(labels):
    labels_matrix[:, idx] = labels_batch[label]

  encoding["labels"] = labels_matrix.tolist()
  
  return encoding

encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)

example = encoded_dataset['train'][0]
print(example.keys())

tokenizer.decode(example['input_ids'])

[id2label[idx] for idx, label in enumerate(example['labels']) if label == 1.0]

encoded_dataset.set_format('torch')

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1",
                                                           problem_type="multi_label_classification",
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)

# Train the model 

batch_size = 8
metric_name = "f1"

from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    f"bert-finetuned-sem_eval-finnish",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    #push_to_hub=True,
)

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch
    
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result


encoded_dataset['train'][0]['labels'].type()

encoded_dataset['train']['input_ids'][0]

import tensorflow as tf

#forward pass
outputs = model(input_ids=encoded_dataset['train']['input_ids'][0].unsqueeze(0), labels=encoded_dataset['train'][0]['labels'].unsqueeze(0))
outputs

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["val"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()

# Test the model on a new sequence 

#text = "Jeg er glad for at jeg endelig kan trene opp en modell for multi-label klassifisering"
text = "Olen iloinen, että voin vihdoin kouluttaa mallin monimerkkiluokitusta varten"

encoding = tokenizer(text, return_tensors="pt")
encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}

outputs = trainer.model(**encoding)

logits = outputs.logits
logits.shape

# apply sigmoid + threshold
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(logits.squeeze().cpu())
predictions = np.zeros(probs.shape)
predictions[np.where(probs >= 0.5)] = 1
# turn predicted id's into actual label names
predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
print(predicted_labels)

