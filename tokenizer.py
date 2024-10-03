from transformers import GPT2Tokenizer
import torch

# Load the cleaned text data
with open("harryPotterData/harry_potter_structured.txt", "r", encoding='utf-8') as file:
    text_data = file.read()

# Initialize the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define a maximum sequence length for each chunk
max_length = 1024  # GPT-2's maximum input length

# Split text into smaller chunks of approximately max_length tokens
# Since the text can be long, we'll use a sliding window approach to avoid overlap issues
text_chunks = [text_data[i:i+max_length] for i in range(0, len(text_data), max_length)]

# Tokenize each chunk and store the tokens
tokenized_chunks = [tokenizer.encode(chunk, max_length=max_length, truncation=True) for chunk in text_chunks]

# Convert the tokenized chunks into tensors
input_tensors = [torch.tensor(chunk) for chunk in tokenized_chunks]

# Save the tensors for training
torch.save(input_tensors, "harry_potter_tokenized.pt")

print(f"Data successfully tokenized into {len(input_tensors)} chunks with each chunk having a maximum of {max_length} tokens.")

