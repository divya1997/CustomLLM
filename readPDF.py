import PyPDF2

def extract_text_from_pdf(pdf_path, output_file):
    # Open the PDF file in binary mode
    with open(pdf_path, 'rb') as file:
        # Create a PDF reader object
        reader = PyPDF2.PdfReader(file)
        # Initialize a variable to hold all the text
        all_text = ""

        # Loop through each page in the PDF
        for page_num in range(len(reader.pages)):
            # Extract text from the page
            page = reader.pages[page_num]
            text = page.extract_text()
            if text:
                # Append text to the variable
                all_text += text

        # Save the extracted text to a file
        with open(output_file, 'w', encoding='utf-8') as text_file:
            text_file.write(all_text)

        print(f"Text successfully extracted and saved to {output_file}")

# Provide the path to the PDF file and desired output file
pdf_path = "harrypotter.pdf"
output_file = "harry_potter_extracted.txt"

# Call the function to extract text
extract_text_from_pdf(pdf_path, output_file)
