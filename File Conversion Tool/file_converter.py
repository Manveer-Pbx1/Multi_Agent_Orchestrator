# HOW TO USE: 
# 1. The program is designed to take the very first file from the files folder and convert it to the desired format.
# 2. Just run `python file_converter.py <extension name (pdf/doc)>`
# 3. The program will convert the file to the desired format and save it in the converted folder.
import os
import fitz  
from docx import Document
from docx.shared import Inches
import win32com.client  

def convert_pdf_to_docx(input_file, output_file):
    """
    Converts a PDF file to a DOCX file while preserving text and images.
    """
    pdf_document = fitz.open(input_file)
    doc = Document()

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        doc.add_paragraph(f"Page {page_num + 1}", style='Heading 1')

        text = page.get_text("text")
        if text.strip():
            doc.add_paragraph(text)

        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_path = f"temp_image.{image_ext}"
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

            doc.add_picture(image_path, width=Inches(5.0))
            os.remove(image_path)

    doc.save(output_file)
    pdf_document.close()

def convert_docx_to_pdf(input_file, output_file):
    """
    Converts a DOCX file to a PDF using Word application.
    This will only work on Windows with Microsoft Word installed.
    """
    input_file = os.path.abspath(input_file)
    output_file = os.path.abspath(output_file)

    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False  

    try:
        doc = word.Documents.Open(input_file)
        
        doc.SaveAs(output_file, FileFormat=17)  
        
        doc.Close()
        print(f"Converted {input_file} to {output_file}.")
    
    except Exception as e:
        print(f"Error converting {input_file} to PDF: {e}")
    
    finally:
        word.Quit()

def main():
    input_dir = "files"
    output_dir = "converted"

    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(input_dir)
    if not files:
        print("No files found in the input directory.")
        return

    for file in files:
        input_path = os.path.join(input_dir, file)

        if os.path.isfile(input_path):
            if file.lower().endswith(".pdf"):
                output_file = os.path.splitext(file)[0] + ".docx"
                output_path = os.path.join(output_dir, output_file)
                print(f"Converting {file} to {output_file}...")
                convert_pdf_to_docx(input_path, output_path)
                print(f"File converted and saved to {output_path}.")

            elif file.lower().endswith(".docx"):
                output_file = os.path.splitext(file)[0] + ".pdf"
                output_path = os.path.join(output_dir, output_file)
                print(f"Converting {file} to {output_file}...")
                convert_docx_to_pdf(input_path, output_path)
                print(f"File converted and saved to {output_path}.")

            else:
                print(f"Unsupported file format: {file}. Only PDF and DOCX files are supported.")
        else:
            print(f"Skipping directory: {file}")


if __name__ == "__main__":
    main()
