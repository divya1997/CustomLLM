import re
import nltk
nltk.download()
from nltk.tokenize import sent_tokenize

def clean_and_structure_text(text):
    # Replace multiple spaces and tabs with a single space
    text = re.sub(r'\s+', ' ', text)

    # Remove unwanted new lines and normalize spaces
    text = re.sub(r'\n+', ' ', text)

    # Replace multiple spaces between words with a single space
    text = re.sub(r'(\s)+', r'\1', text)

    # Remove space before commas and periods
    text = re.sub(r'\s+([,.!?])', r'\1', text)

    # Sentence tokenize using nltk
    sentences = sent_tokenize(text)

    # Create a formatted version where each sentence is on a new line
    structured_text = "\n".join(sentences)

    return structured_text

# Read the extracted text from the file
with open("harry_potter_extracted.txt", "r", encoding='utf-8') as file:
    raw_text = file.read()

# Clean and structure the text
structured_text = clean_and_structure_text(raw_text)

# Save the structured text to a new file
with open("harry_potter_structured.txt", "w", encoding='utf-8') as structured_file:
    structured_file.write(structured_text)

print("Text cleaned and structured into multiple lines. Saved to harry_potter_structured.txt")
