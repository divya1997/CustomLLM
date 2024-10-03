import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer  # Import tokenizer from Hugging Face library

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Hyperparameters
vocab_size = len(tokenizer)
embed_dim = 128
num_heads = 2
num_layers = 2
learning_rate = 0.001
num_epochs = 5
max_length = 512  # Maximum length of input sequences

# Define the Transformer Encoder model
class SmallTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super(SmallTransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        transformer_output = self.transformer(embedded)
        output = self.fc(transformer_output)
        return output

# Load the data
train_data = torch.load("train_data.pt")

# Initialize the model
model = SmallTransformerModel(vocab_size, embed_dim, num_heads, num_layers)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in train_data:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch)

        # Calculate loss (shift outputs and batch for next-token prediction)
        loss = criterion(outputs.view(-1, vocab_size), batch.view(-1))
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss/len(train_data)}")

# Save the trained model
torch.save(model.state_dict(), "small_transformer_model.pth")
