import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from preprocess import getDataSets
import torch.optim as optim

from transformers import BertTokenizer, BertTokenizerFast



class GRU_QA(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256, num_layers=2):
        super(GRU_QA, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        
        self.start_fc = nn.Linear(hidden_dim * 2, 1)  # Start index prediction
        self.end_fc = nn.Linear(hidden_dim * 2, 1)  # End index prediction
        
    def forward(self, context, question):
        # Embedding lookup
        context_embedded = self.embedding(context)  # (batch_size, max_length, embedding_dim)
        question_embedded = self.embedding(question)

        # Pass through GRU
        context_output, _ = self.gru(context_embedded)  # (batch_size, max_length, hidden_dim*2)
        question_output, _ = self.gru(question_embedded)

        # Predict start & end positions
        start_logits = self.start_fc(context_output).squeeze(-1)  # (batch_size, max_length)
        end_logits = self.end_fc(context_output).squeeze(-1)  # (batch_size, max_length)

        return start_logits, end_logits



class SquadDataset(Dataset):
    ''' PyTorch Dataset for SQuAD v2.0 '''
    def __init__(self, df):
        self.contexts = torch.stack(df['Context'].tolist())  # (num_samples, max_length)
        self.questions = torch.stack(df['Question'].tolist())  # (num_samples, max_length)
        self.start_pos = torch.tensor(df['Start_Pos'].tolist())  # (num_samples,)
        self.end_pos = torch.tensor(df['End_Pos'].tolist())  # (num_samples,)
        self.is_impossible = torch.tensor(df['Is_Impossible'].tolist(), dtype=torch.float)  # (num_samples,)

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return {
            'context': self.contexts[idx],   # Tensor (max_length,)
            'question': self.questions[idx], # Tensor (max_length,)
            'start_pos': self.start_pos[idx],   # Scalar tensor
            'end_pos': self.end_pos[idx],       # Scalar tensor
            'is_impossible': self.is_impossible[idx]  # Scalar tensor (0 or 1)
        }
        
def get_dataloader(train=True, batch_size=16, shuffle=True):
    df = getDataSets(train) 
    dataset = TensorDataset(df["Context"].tolist(),
                            df["Question"].tolist(), 
                            torch.tensor(df["Start_Pos"].tolist()), 
                            torch.tensor(df["End_Pos"].tolist()))

    # dataset = SquadDataset(df)  
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def train(model, epochs, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            context, question, start_pos, end_pos = batch

            optimizer.zero_grad()
            start_logits, end_logits = model(context, question)

            # Compute loss
            loss_start = criterion(start_logits, start_pos)
            loss_end = criterion(end_logits, end_pos)
            loss = (loss_start + loss_end) / 2  # Average loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')

    print('Training complete!')

def main(): 
    train_loader = get_dataloader()
    # test_loader = get_dataloader(train=False)
    
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    
    # model = GRU_QA(vocab_size=tokenizer.vocab_size)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    # train(model, 3, train_loader, optimizer, criterion)
    
    for batch in train_loader:
        # print('Context shape:', batch['context'].shape)  # (batch_size, max_length)
        # print('Question shape:', batch['question'].shape)  # (batch_size, max_length)
        # print('Start positions:', batch['start_pos'])  # (batch_size,)
        # print('End positions:', batch['end_pos'])  # (batch_size,)
        # print('Is Impossible:', batch['is_impossible'])  # (batch_size,)
        print(batch)
        break  # Stop after one batch

# if __name__ == '__main__':
#     main()

def train_gru(epochs=3, batch_size=16, lr=1e-3):
    # Load dataset
    print("Loading dataset...")
    df = getDataSets(train=True)

    # # Convert dataset to DataLoader
    # dataset = TensorDataset(df["Context"].tolist(), df["Question"].tolist(), 
    #                         torch.tensor(df["Start_Pos"].tolist()), 
    #                         torch.tensor(df["End_Pos"].tolist()))
    # Correct TensorDataset construction for GRU
    context_tensor = torch.stack(df["Context"].tolist())  # Ensure context is a tensor
    question_tensor = torch.stack(df["Question"].tolist())  # Ensure question is a tensor
    start_pos_tensor = torch.tensor(df["Start_Pos"].tolist())  # Ensure start position is a tensor
    end_pos_tensor = torch.tensor(df["End_Pos"].tolist())  # Ensure end position is a tensor

    dataset = TensorDataset(context_tensor, question_tensor, start_pos_tensor, end_pos_tensor)

    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = GRU_QA(vocab_size=tokenizer.vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    print("Starting training...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            context, question, start_pos, end_pos = batch

            optimizer.zero_grad()
            start_logits, end_logits = model(context, question)

            # Compute loss
            loss_start = criterion(start_logits, start_pos)
            loss_end = criterion(end_logits, end_pos)
            loss = (loss_start + loss_end) / 2  # Average loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

    print("Training complete!")
    torch.save(model.state_dict(), "gru_model.pth")
    print("Model saved!")
    
from torch.utils.data import DataLoader, TensorDataset

def evaluate_gru(batch_size=16):
    # Load dataset
    print("Loading validation dataset...")
    df = getDataSets(train=False)  # Load dev set for evaluation

    # Convert dataset to DataLoader
    context_tensor = torch.stack(df["Context"].tolist())
    question_tensor = torch.stack(df["Question"].tolist())
    start_pos_tensor = torch.tensor(df["Start_Pos"].tolist())
    end_pos_tensor = torch.tensor(df["End_Pos"].tolist())

    dataset = TensorDataset(context_tensor, question_tensor, start_pos_tensor, end_pos_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize model and load trained weights
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = GRU_QA(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load("gru_model.pth"))  # Assuming the model is saved with this name
    model.eval()

    total_correct_start = 0
    total_correct_end = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradients during evaluation
        for batch in dataloader:
            context, question, start_pos, end_pos = batch

            # Get model predictions
            start_logits, end_logits = model(context, question)

            # Get the predicted start and end positions
            start_preds = torch.argmax(start_logits, dim=1)
            end_preds = torch.argmax(end_logits, dim=1)

            # Calculate correct predictions
            total_correct_start += (start_preds == start_pos).sum().item()
            total_correct_end += (end_preds == end_pos).sum().item()
            total_samples += len(start_pos)

    # Calculate accuracy for start and end positions
    start_accuracy = total_correct_start / total_samples
    end_accuracy = total_correct_end / total_samples
    print(f"Start Position Accuracy: {start_accuracy * 100:.2f}%")
    print(f"End Position Accuracy: {end_accuracy * 100:.2f}%")


if __name__ == "__main__":
    # Call training and evaluation
    train_gru()  # Train the model first
    evaluate_gru()  # Then evaluate it



if __name__ == "__main__":
    train_gru()