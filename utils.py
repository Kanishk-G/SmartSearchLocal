import os
from pypdf import PdfReader
from docx import Document
from reportlab.pdfgen.canvas import Canvas
from datasets import load_dataset


class TextExtractor:
    def __init__(self) -> None:

        pass

    def get_file_extension(self, filepath :str):
        return os.path.splitext(filepath)[-1]

    def read_text(self, filepath):
        extension = self.get_file_extension(filepath)

        if extension == ".txt":
            with open(filepath, "rb") as f:
                file_content = f.read()
                return file_content.decode('utf-8')
            
        elif extension == ".pdf":
            pdf_reader = PdfReader(filepath)
            file_content = ""
            for page in pdf_reader.pages:
                file_content += page.extract_text()
                return file_content
            
        elif extension == ".docx":
            doc = Document(filepath)
            file_content = ""
            for paragraph in doc.paragraphs:
                file_content += paragraph.text + "\n"
            return file_content

    
    def get_sample_data(self):
        dataset = list(load_dataset("wikipedia", "20220301.simple")["train"])

        dataset = dataset[:10]

        return dataset
    
    def create_sample_pdf(self):
        dataset = self.get_sample_data()[0:2]

        for data in dataset:
            canvas = Canvas(f'sampleFiles/{data["title"]}.pdf')
            canvas.drawString(72, 792, data['text'])
            canvas.save()
    
    def create_sample_txt(self):
        dataset = self.get_sample_data()
        for data in dataset:
            with open(f'../sampleFiles/{data["title"]}.txt', 'w', encoding='utf-8') as f:
                f.write(data["text"])
        
if __name__ == "__main__":
    extractor = TextExtractor()
    extractor.create_sample_txt()




