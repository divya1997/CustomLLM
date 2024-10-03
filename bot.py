import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load your model class definition
# from your_model_file import SmallTransformerModel
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

    def generate(self, input_ids, max_length=50):
        self.eval()  # Set the model to evaluation mode

        # Start with the input ids
        generated = input_ids.clone()
        
        for _ in range(max_length):
            with torch.no_grad():  # No need to calculate gradients
                # Forward pass to get predictions
                outputs = self.forward(generated)

                # Take the last token's predictions
                next_token_logits = outputs[:, -1, :]
                
                # Get the predicted token ID (greedy search)
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                
                # Append the predicted token to the generated sequence
                generated = torch.cat((generated, next_token), dim=1)

        return generated
    
    def top_k_sampling(self, logits, top_k=50):
        # Get the top k logits and their indices
        indices_to_keep = logits.topk(top_k).indices
        filtered_logits = torch.full(logits.shape, float('-inf'))
        filtered_logits.scatter_(1, indices_to_keep, logits.gather(1, indices_to_keep))

        return filtered_logits

    #############
    def generate_temp(self, input_ids, max_length=50, temperature=1.0):
        self.eval()  # Set the model to evaluation mode

        generated = input_ids.clone()
        
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.forward(generated)
                next_token_logits = outputs[:, -1, :]
            
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Use softmax to get probabilities and sample from them
                probabilities = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1).squeeze(1)  # Corrected this line
                
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)  # Ensure correct shape

        return generated

    def generate_samp(self, input_ids, max_length=50, top_k=50):
        self.eval()

        generated = input_ids.clone()
        
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.forward(generated)
                next_token_logits = outputs[:, -1, :]
                
                # Apply top-k sampling
                next_token_logits = self.top_k_sampling(next_token_logits, top_k)
                
                # Sample from the filtered logits
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)  # Corrected this line
                
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)  # Ensure correct shape

        return generated




# Initialize the model
vocab_size = len(tokenizer)  # Set this to your actual vocab size
embed_dim = 128
num_heads = 2
num_layers = 2

model = SmallTransformerModel(vocab_size, embed_dim, num_heads, num_layers)
model.load_state_dict(torch.load("small_transformer_model.pth"))
model.eval()  # Set the model to evaluation mode

# Sample input text
input_text = "Harry Potter is a"

# Tokenize and encode the input text
input_ids = tokenizer.encode(input_text, return_tensors='pt')  # Shape: (1, seq_len)


# Generate text using greedy search
output = model.generate(input_ids, max_length=50)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Greedy Search Output:\n", generated_text)

# Generate text using temperature sampling
output = model.generate_temp(input_ids, max_length=50, temperature=0.8)  # Experiment with temperature
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Temperature Sampling Output:\n", generated_text)

# Generate text using top-k sampling
output = model.generate_samp(input_ids, max_length=50, top_k=50)  # Experiment with top_k
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Top-k Sampling Output:\n", generated_text)





