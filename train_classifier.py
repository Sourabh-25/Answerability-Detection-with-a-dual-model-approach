from transformers import BertTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch
from models import QAClassifier
from datasets import load_dataset

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = QAClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Dataset class
class AnswerableDataset(torch.utils.data.Dataset):
    def __init__(self, split='train'):
        data = load_dataset('squad_v2', split=split)
        self.samples = []
        for item in data:
            inputs = tokenizer(item['question'], item['context'],
                               truncation='only_second',
                               padding='max_length',
                               max_length=384,
                               return_tensors='pt')
            label = 0 if len(item['answers']['text']) == 0 else 1
            self.samples.append((inputs, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inputs, label = self.samples[idx]
        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze(), torch.tensor(label)

# Collate function
def collate_fn(batch):
    input_ids = torch.stack([x[0] for x in batch])
    attention_mask = torch.stack([x[1] for x in batch])
    labels = torch.stack([x[2] for x in batch])
    return input_ids, attention_mask, labels

# DataLoader
train_ds = AnswerableDataset('train[:100%]')  # Use small subset for quick training
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
# model.train()
cnt=0
num_epochs = 3
num_batches = len(train_dl)
total_iterations = num_epochs * num_batches
print(f"Total iterations: {total_iterations}")

for epoch in range(3):
    total_loss = 0
    for input_ids, attention_mask, labels in train_dl:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        logits = model(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cnt+=1
        total_loss += loss.item()
        if(cnt%100==0):
            print(f"itr= {cnt}")
        print(f"loss: {loss}")
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "qa_classifier_model.pt")
print(" Trained classifier model saved to 'qa_classifier_model.pt'")


