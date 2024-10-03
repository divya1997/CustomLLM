# CustomLLM
Building a custom LLM from scratch
- The core logic for this project is in train.py. This holds both the transformer class and training algorithm.
- This provides a solid groundwork for building bigger models.
- This is quite a small model, and the generated output from the Transformer is gibberish.

TODO:
- The next step is to train and fine-tune an already available model(mostly GPT2).
- Improve ways to structure the data.
- Implement the transformer algo in C++, instead of Python.

List of file and their utility to build and train the LLM.
1. ./harryPotterData/readPDF.py-------------------//read the PDF file and store data into txt file.
2. ./harryPotterData/cleanHarryPotterData.py------//Remove unwanted spaces and tabs.
3. ./harryPotterData/structureHarryPotter.py------//Make data in structured format, and create harry_potter_structured.txt file.
4. ./harryPotterData/harry_potter_structured.txt    
5. ./tokenizer.py---------------------------------//create tokens from data
6. ./splitData.py---------------------------------//create train_data and val_data
7. ./train.py-------------------------------------//Transformer model, and training of model
8. ./validateModel.py-----------------------------//validate the model loss
8. ./bot.py---------------------------------------//generate ans using model
9. ./harry_potter_tokenized.pt--------------------//created by tokenizer
    - ./train_data.py
    - ./val_date.py
10. ./small_transformer_model.pth-----------------//model
