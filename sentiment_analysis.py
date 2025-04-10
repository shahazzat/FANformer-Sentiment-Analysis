# OLD (deprecated in newer transformers):
# from transformers import AdamW

# NEW (use PyTorch's native AdamW):
from torch.optim import AdamW

# 1. Install dependencies
!pip install torch transformers datasets

# 2. Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from transformers import AutoTokenizer

# 3. Define Linear Attention (FANformer's core)
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)

        # Linear attention with ELU kernel
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Compute attention (linear complexity)
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        z = 1 / (torch.einsum('bhnd,bhd->bhn', q, k.sum(dim=2)) + 1e-6)  # +epsilon for stability
        out = torch.einsum('bhnd,bhde,bhn->bhne', q, kv, z)
        return self.to_out(out.transpose(1, 2).reshape(b, n, -1))
		
# 4. Build FANformer Model
class FANformer(nn.Module):
    # def __init__(self, vocab_size, dim=256, heads=8, num_layers=6):
    def __init__(self, vocab_size, dim=264, heads=11, num_layers=9): # Ensure dim % heads == 0
        assert dim % heads == 0, "Embedding dimension (dim) must be divisible by heads"
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                LinearAttention(dim, heads),
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            ) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(dim, 2)  # For binary classification

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
        return self.classifier(x.mean(dim=1))  # Mean pooling

# 5. Load IMDb Dataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("imdb")

def encode(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

dataset = dataset.map(encode, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
train_loader = DataLoader(dataset["train"].select(range(1000)), batch_size=16, shuffle=True)  # Smaller subset for demo

# 6. Initialize Model and Optimizer (CORRECT ORDER)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FANformer(tokenizer.vocab_size).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)  # Now model is defined!


from torch.utils.data import random_split

# 7. Split dataset (70% train, 15% val, 15% test)
train_size = int(0.7 * len(dataset["train"]))
val_size = (len(dataset["train"]) - train_size) // 2
test_size = len(dataset["train"]) - train_size - val_size

train_data, val_data, test_data = random_split(
    dataset["train"], [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)  # For reproducibility
)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)
test_loader = DataLoader(test_data, batch_size=16)

def validate(model, dataloader):
    model.eval()
    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)

            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()

    accuracy = total_correct / len(dataloader.dataset)
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy

for epoch in range(3):
    # Training
    model.train()
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    val_loss, val_acc = validate(model, val_loader)

    print(
        f"Epoch {epoch} | "
        f"Train Loss: {train_loss/len(train_loader):.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc*100:.2f}%"
    )

# 8. Testing (Final Evaluation)
test_loss, test_acc = validate(model, test_loader)
print(f"\nTest Accuracy: {test_acc*100:.2f}% | Test Loss: {test_loss:.4f}")

# 9. Inference on New Text
def predict(text, model, tokenizer):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model(inputs["input_ids"])
        print(logits)
        print(logits.shape) # Should match (batch_size, num_classes)
        print(tokenizer.decode(inputs["input_ids"][0]))  # Ensure correct input text
    prob = F.softmax(logits, dim=1)
    print(prob)
    print(torch.argmax(prob).item())
    return "Positive" if torch.argmax(prob).item() == 1 else "Negative"

# Example
sample_review = "This movie was fantastic! The acting blew me away."
# sample_review = "very good!!"
print(predict(sample_review, model, tokenizer))  # Output: "Positive"

torch.save(model.state_dict(), "best_fanformer.pth")