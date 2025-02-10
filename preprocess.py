import json
import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset

def json_to_df(train=True):
    if train == False:
        with open('Data/dev-v2.0.json', 'r') as f:
            squad_data = json.load(f)
    else:        
        with open('Data/train-v2.0.json', 'r') as f:
                squad_data = json.load(f)

    data = []

    for group in squad_data['data']:
        title = group['title']
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                is_impossible = qa['is_impossible']
                
                answer = qa['answers'][0]['text'] if qa['answers'] else ""

                data.append({
                    "Title": title,
                    "Context": context,
                    "Question": question,
                    "Answer": answer,
                    "Is_Impossible": is_impossible
                })

    df = pd.DataFrame(data)
    return df


def tokenize_data(df, tokenizer, max_length=512):
    df['Context'] = df['Context'].apply(lambda x: tokenizer.encode(x, max_length=max_length, truncation=True, padding='max_length'))
    df['Question'] = df['Question'].apply(lambda x: tokenizer.encode(x, max_length=max_length, truncation=True, padding='max_length'))
    df['Answer'] = df['Answer'].apply(lambda x: tokenizer.encode(x, max_length=max_length, truncation=True, padding='max_length'))
    return df

def convert_to_tensors(df):
    df['Context'] = df['Context'].apply(lambda x: torch.tensor(x))
    df['Question'] = df['Question'].apply(lambda x: torch.tensor(x))
    df['Answer'] = df['Answer'].apply(lambda x: torch.tensor(x))
    return df

class SquadDataset(Dataset):
    def __init__(self, df):
        self.contexts = df['Context'].tolist()
        self.questions = df['Question'].tolist()
        self.answers = df['Answer'].tolist()
        self.is_impossible = df['Is_Impossible'].tolist()

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return {
            'context': self.contexts[idx],
            'question': self.questions[idx],
            'answer': self.answers[idx],
            'is_impossible': self.is_impossible[idx]
        }

def getDataSets(train = True):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    df = json_to_df(train)
    df = tokenize_data(df, tokenizer)
    df = convert_to_tensors(df)
    
    return SquadDataset(df)
    
if __name__ == '__main__':
    getDataSets()