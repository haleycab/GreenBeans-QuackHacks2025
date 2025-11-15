# Convert a PDF to HTML text 
# Source: Datacamp An Introduction to Python Subprocess: Basics and Examples 


# Get it working in terminal 
import subprocess
import os

def PDFtoText(pdf):
    pieces = pdf.split('/').pop(-1).split('.') # Returns an array of strings 
    pieces.pop(-1) 
    name = ".".join(pieces) #original name back without pdf
    result = subprocess.run(["pdftotext", pdf, f"outputs/{name}.txt"], capture_output=True, text=True) 
    print(result.stdout)
    print(result.stderr)


inputs = os.listdir("inputs")
for i in inputs:
    PDFtoText(f"inputs/{i}")
    test = f"inputs/{i}"
    print(test)


