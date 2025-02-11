import json
import pandas as pd
import torch
from transformers import BertTokenizer, BertTokenizerFast
from torch.utils.data import Dataset, DataLoader

def json_to_df(train=True):
    """Load SQuAD v2.0 JSON and convert to DataFrame"""
    file_path = 'Data/dev-v2.0.json' if not train else 'Data/train-v2.0.json'
    
    with open(file_path, 'r') as f:
        squad_data = json.load(f)

    data = []

    for group in squad_data['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                is_impossible = qa['is_impossible']
                
                if qa['answers']:
                    answer_text = qa['answers'][0]['text']
                    answer_start = qa['answers'][0]['answer_start']
                    answer_end = answer_start + len(answer_text)
                else:
                    answer_text = ""
                    answer_start, answer_end = -1, -1

                data.append({
                    "Context": context,
                    "Question": question,
                    "Answer": answer_text,
                    "Start_Pos": answer_start,
                    "End_Pos": answer_end,
                    "Is_Impossible": is_impossible
                })

    return pd.DataFrame(data)


def tokenize_data(df, tokenizer, max_length=512):
    """Tokenizes SQuAD data and maps answer start/end positions"""
    tokenized_data = {
        "Context": [],
        "Question": [],
        "Start_Pos": [],
        "End_Pos": [],
        "Is_Impossible": []
    }

    for _, row in df.iterrows():
        context = row["Context"]
        question = row["Question"]
        start_char_idx = row["Start_Pos"]

        context_encoding = tokenizer.encode(context, truncation=True, padding="max_length", max_length=max_length)
        question_encoding = tokenizer.encode(question, truncation=True, padding="max_length", max_length=max_length)

        start_token, end_token = find_token_indices(tokenizer, context, row["Answer"], start_char_idx)

        tokenized_data["Context"].append(context_encoding)
        tokenized_data["Question"].append(question_encoding)
        tokenized_data["Start_Pos"].append(start_token)
        tokenized_data["End_Pos"].append(end_token)
        tokenized_data["Is_Impossible"].append(row["Is_Impossible"])

    return pd.DataFrame(tokenized_data)



def find_token_indices(tokenizer, context, answer, start_char_idx):
    """Finds start and end token indices in the tokenized context"""
    if start_char_idx == -1:  # No answer case
        return 0, 0

    # Tokenize context with word-to-token mapping
    encoding = tokenizer(context, return_offsets_mapping=True, truncation=True)
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]  # List of (start_char, end_char) for each token

    start_token_idx = None
    end_token_idx = None

    for i, (char_start, char_end) in enumerate(offsets):
        if char_start <= start_char_idx < char_end:
            start_token_idx = i
        if char_start < start_char_idx + len(answer) <= char_end:
            end_token_idx = i
            break

    if start_token_idx is None or end_token_idx is None:
        return 0, 0  # Default to 0 if we fail to find positions

    return start_token_idx, end_token_idx



def convert_to_tensors(df):
    """Converts tokenized lists to PyTorch tensors"""
    df["Context"] = df["Context"].apply(lambda x: torch.tensor(x))
    df["Question"] = df["Question"].apply(lambda x: torch.tensor(x))
    return df


def save_torch_dataset(df, path):
    """Saves preprocessed dataset as a PyTorch tensor file"""
    torch.save(df.to_dict(), path)


def getDataSets(train=True):
    """Loads preprocessed dataset or preprocesses from scratch"""
    path = f'preprocessed_data_gru_lstm_{"train" if train else "dev"}.pt'
    
    try:
        loaded_data = torch.load(path)
        loaded_df = pd.DataFrame(loaded_data)
    except FileNotFoundError:
        print("Preprocessed data not found. Processing...")
        df = json_to_df(train)
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        df = tokenize_data(df, tokenizer)
        df = convert_to_tensors(df)
        save_torch_dataset(df, path)
        loaded_df = df  

    return loaded_df


if __name__ == '__main__':
    dataset = getDataSets()
    print("Dataset loaded successfully!")
    print(f"Number of samples: {len(dataset)}")
