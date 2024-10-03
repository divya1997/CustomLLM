from sklearn.model_selection import train_test_split
import torch
# Load the tokenized data
input_tensors = torch.load("harry_potter_tokenized.pt")

# Split the dataset
train_data, val_data = train_test_split(input_tensors, test_size=0.1)

# Save the split data
torch.save(train_data, "train_data.pt")
torch.save(val_data, "val_data.pt")

print("Data split into training and validation sets.")
