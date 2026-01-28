import json
import os
import subprocess

input_file = ("results1.json")
output_dir = "pdf_out1"
batch_size = 10
os.makedirs(output_dir, exist_ok=True)

results = []



with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

    for i, ex in enumerate(data):
        messages = ex["messages"]
        generated_text = ex["generated"].strip()
        n=0
        for m in messages:
            content=m["content"]
            n+=len(content)
        clean_generated = generated_text[n:].strip()
        results.append(clean_generated)

for batch_idx in range(1, len(results), batch_size):
    batch = results[batch_idx:batch_idx + batch_size]

    md_content = f"# Batch {batch_idx // batch_size + 1}\n\n"
    for j, text in enumerate(batch, start=1):
        md_content += f"## Generated {batch_idx + j}\n\n"
        md_content += text + "\n\n---\n\n"

    md_path = os.path.join(output_dir, f"file_{batch_idx // batch_size + 1}.md")
    pdf_path = os.path.join(output_dir, f"file_{batch_idx // batch_size + 1}.pdf")

    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(md_content)
    pandoc_path = r"C:\Users\uraa-\AppData\Local\Pandoc\pandoc.exe"
    subprocess.run([
        pandoc_path, md_path, "-o", pdf_path,
        "--pdf-engine=xelatex",
        "-V", "mainfont=Times New Roman"
    ], check=True)

    print(f"Готов PDF: {pdf_path}")
