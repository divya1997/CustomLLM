import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer  # Import tokenizer from Hugging Face library

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define your SmallTransformerModel class here
# ...Define the Transformer Encoder model
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

# Load the model
vocab_size = len(tokenizer)  # Ensure you define or load the tokenizer before this line
embed_dim = 128  # Or the value you used during training
num_heads = 2     # Or the value you used during training
num_layers = 2    # Or the value you used during training

model = SmallTransformerModel(vocab_size, embed_dim, num_heads, num_layers)
model.load_state_dict(torch.load("small_transformer_model.pth"))
model.eval()  # Set the model to evaluation mode

# Evaluation function
def evaluate(model, val_data, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    total_tokens = 0

    with torch.no_grad():  # No need to calculate gradients during evaluation
        for batch in val_data:
            outputs = model(batch)
            loss = criterion(outputs.view(-1, vocab_size), batch.view(-1))

            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=-1)
            correct_predictions += (predictions == batch).sum().item()
            total_tokens += batch.numel()

    avg_loss = total_loss / len(val_data)
    accuracy = correct_predictions / total_tokens
    return avg_loss, accuracy

# Load validation data
val_data = torch.load("val_data.pt")  # Assuming you have a separate validation file

# Define your loss function
criterion = nn.CrossEntropyLoss()  # Adjust based on your model's output

# Evaluate on validation set
val_loss, val_accuracy = evaluate(model, val_data, criterion)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")


