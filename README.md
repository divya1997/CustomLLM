---

# **CustomLLM**

### **Building a Custom Language Model from Scratch**

This project focuses on building a small-scale Language Model (LLM) from scratch using Transformer architectures. The core logic is implemented in the `train.py` file, which includes both the custom Transformer class and the training algorithm. While this implementation is a basic version, it serves as a solid foundation for building and experimenting with larger models.

### **Project Overview**

This model was trained on a relatively small dataset using text from the **Harry Potter** book series. Due to the limited training data, the current model is prone to overfitting and tends to produce repetitive or nonsensical outputs. Nonetheless, this project showcases how to build a Transformer-based LLM from scratch.

### **Key Features:**
1. **End-to-End Custom Implementation**: From data preprocessing to tokenization, model training, and text generation.
2. **Minimalistic Transformer Architecture**: Uses a simple yet effective approach to implement the Transformer structure.
3. **Training and Validation Framework**: Implements basic training and validation scripts for tracking model performance.

### **TODO:**
1. **Fine-tune Pre-trained Models**: Use existing models (e.g., GPT-2) and adapt them for specific tasks to improve the quality of generated content.
2. **Improve Data Structuring**: Enhance the preprocessing steps to ensure cleaner input for the model.
3. **Implement the Transformer in C++**: Build a high-performance version of the Transformer module using C++ for faster inference and lower resource usage.

### **File Structure and Usage:**
| File / Directory                                  | Description                                                                 |
|--------------------------------------------------|-----------------------------------------------------------------------------|
| `./harryPotterData/readPDF.py`                    | Reads the PDF files and stores the data into a structured text file.        |
| `./harryPotterData/cleanHarryPotterData.py`       | Cleans the text by removing unwanted spaces, tabs, and formatting issues.   |
| `./harryPotterData/structureHarryPotter.py`       | Structures the cleaned data and saves it as `harry_potter_structured.txt`.  |
| `./harryPotterData/harry_potter_structured.txt`   | The structured text data used for training.                                 |
| `./tokenizer.py`                                  | Creates tokenized representations of the text data.                         |
| `./splitData.py`                                  | Splits the data into training and validation datasets.                      |
| `./train.py`                                      | Defines the custom Transformer model and handles training logic.            |
| `./validateModel.py`                              | Validates the trained model and computes the loss on the validation set.    |
| `./bot.py`                                        | Generates text responses using the trained model.                           |
| `./harry_potter_tokenized.pt`                     | Serialized tokenized data created by the tokenizer.                         |
| `./train_data.py`                                 | Training data generated from the dataset.                                   |
| `./val_data.py`                                   | Validation data generated from the dataset.                                 |
| `./small_transformer_model.pth`                   | Saved model weights for the small Transformer model.                        |

### **How to Use:**
1. **Preprocess the Dataset**:
   - Run the scripts in the `harryPotterData` directory sequentially to extract, clean, and structure the dataset.
2. **Tokenize the Text**:
   - Use `tokenizer.py` to convert the structured text into token IDs.
3. **Split the Data**:
   - Use `splitData.py` to create training and validation splits.
4. **Train the Model**:
   - Execute `train.py` to train the custom Transformer model.
5. **Validate the Model**:
   - Use `validateModel.py` to monitor the model's loss and performance.
6. **Generate Text**:
   - Use `bot.py` to generate text responses from the trained model.

### **Next Steps:**
1. Continue fine-tuning using larger, pre-trained models like GPT-2.
2. Extend the model architecture for more complex tasks and larger datasets.
3. Implement a C++ version for high-efficiency production use.

Feel free to explore and contribute to this project. The goal is to build a compact yet effective LLM that can run on low-resource environments such as mobile devices.

--- 

