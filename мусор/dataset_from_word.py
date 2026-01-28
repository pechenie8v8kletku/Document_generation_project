import subprocess
import os

finecmd = r"C:\Program Files (x86)\ABBYY FineReader 14\FineCmd.exe"
def process(input_file,output_file):
    subprocess.run([
        finecmd,
        input_file,
        "/out", output_file,
        "/lang", "Russian","English",
        "/format", "docx",
        "/quiet"
    ])
input_folder="downloaded_gosts"
files=os.listdir(input_folder)
output_folder = "word from download"
for a in files:
    file_path = os.path.join(input_folder, a)
    output_name = os.path.join(output_folder,os.path.splitext(a)[0]) + ".docx"
    process(file_path,output_name)

