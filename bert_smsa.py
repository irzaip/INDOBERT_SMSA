import toml
import os, sys
#sys.path.append('../')
#os.chdir('../')
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import random
import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
import helper

#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#from sklearn.linear_model import LogisticRegression
#import fasttext
#from nltk import word_tokenize

from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
#from nltk.tokenize import TweetTokenizer

from utils.forward_fn import forward_sequence_classification
from utils.metrics import document_sentiment_metrics_fn
from utils.data_utils import DocumentSentimentDataset, DocumentSentimentDataLoader

cfg = toml.load('bert_smsa.toml')
num_workers = cfg['CONFIG']['num_workers']
batch_size = cfg['CONFIG']['batch_size']
max_seq_len = cfg['CONFIG']['max_seq_len']
n_epochs = cfg['CONFIG']['n_epochs']

train_dataset_path = cfg['CONFIG']['train_dataset_path']
valid_dataset_path = cfg['CONFIG']['valid_dataset_path']
test_dataset_path = cfg['CONFIG']['test_dataset_path']


# Load Tokenizer and Config
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
config = BertConfig.from_pretrained('indobenchmark/indobert-base-p1')
config.num_labels = DocumentSentimentDataset.NUM_LABELS

# Instantiate model
model = BertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p1', config=config)

model.load_state_dict(torch.load('smsa.model'))



def predict(input: str) -> str :
    subwords = tokenizer.encode(input)
    subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

    logits = model(subwords)[0]
    label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()

    print(f'Text: {input} | Label : {i2w[label]} ({F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}%)')
    return '{' + f'sentiment: {i2w[label]}  score: {F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}%' + '}'

def prepare_data():
    global train_loader, valid_loader, test_loader, w2i, i2w, optimizer, model
    train_dataset = DocumentSentimentDataset(train_dataset_path, tokenizer, lowercase=True)
    valid_dataset = DocumentSentimentDataset(valid_dataset_path, tokenizer, lowercase=True)
    test_dataset = DocumentSentimentDataset(test_dataset_path, tokenizer, lowercase=True)

    train_loader = DocumentSentimentDataLoader(dataset=train_dataset, max_seq_len=max_seq_len, batch_size=batch_size, num_workers=num_workers, shuffle=True)  
    valid_loader = DocumentSentimentDataLoader(dataset=valid_dataset, max_seq_len=max_seq_len, batch_size=batch_size, num_workers=num_workers, shuffle=False)  
    test_loader = DocumentSentimentDataLoader(dataset=test_dataset, max_seq_len=max_seq_len, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    w2i, i2w = DocumentSentimentDataset.LABEL2INDEX, DocumentSentimentDataset.INDEX2LABEL

    optimizer = optim.Adam(model.parameters(), lr=5e-6)
    model = model.cuda()

def train():
    for epoch in range(n_epochs):
        model.train()
        torch.set_grad_enabled(True)
    
        total_train_loss = 0
        list_hyp, list_label = [], []

        train_pbar = tqdm(train_loader, leave=True, total=len(train_loader))
        for i, batch_data in enumerate(train_pbar):
            # Forward model
            loss, batch_hyp, batch_label = forward_sequence_classification(model, batch_data[:-1], i2w=i2w, device='cuda')

            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss = loss.item()
            total_train_loss = total_train_loss + tr_loss

            # Calculate metrics
            list_hyp += batch_hyp
            list_label += batch_label

            train_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format((epoch+1),
                total_train_loss/(i+1), helper.get_lr(optimizer)))

        # Calculate train metric
        metrics = document_sentiment_metrics_fn(list_hyp, list_label)
        print("(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format((epoch+1),
            total_train_loss/(i+1), helper.metrics_to_string(metrics), helper.get_lr(optimizer)))

        # Evaluate on validation
        model.eval()
        torch.set_grad_enabled(False)
        
        total_loss, total_correct, total_labels = 0, 0, 0
        list_hyp, list_label = [], []

        pbar = tqdm(valid_loader, leave=True, total=len(valid_loader))
        for i, batch_data in enumerate(pbar):
            batch_seq = batch_data[-1]        
            loss, batch_hyp, batch_label = forward_sequence_classification(model, batch_data[:-1], i2w=i2w, device='cuda')
            
            # Calculate total loss
            valid_loss = loss.item()
            total_loss = total_loss + valid_loss

            # Calculate evaluation metrics
            list_hyp += batch_hyp
            list_label += batch_label
            metrics = document_sentiment_metrics_fn(list_hyp, list_label)

            pbar.set_description("VALID LOSS:{:.4f} {}".format(total_loss/(i+1), helper.metrics_to_string(metrics)))
            
        metrics = document_sentiment_metrics_fn(list_hyp, list_label)
        print("(Epoch {}) VALID LOSS:{:.4f} {}".format((epoch+1),
            total_loss/(i+1), helper.metrics_to_string(metrics)))
        
if __name__ == '__main__':
    prepare_data()
    #print(w2i,i2w)
    #train()
    predict("hello banding")
