from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
import os
from мусор.ziga import append_to_json_file


class SDTParagraph(Paragraph):
    def __init__(self, element, parent):
        super().__init__(element, parent)
        self.is_sdt = True

input_folder="word from download by docx microsoft"
files=os.listdir(input_folder)
json_path = "../just_test.json"
MAXLEN=0
def iter_block_items(parent):
    def yield_paragraph(p_el, parent):
        para = Paragraph(p_el, parent)
        try:
            style = para.style.name.lower()
            if "заголовок" in style or "heading" in style:
                return SDTParagraph(p_el, parent)
        except:
            pass
        return para

    def inner_iter(el, parent):
        for child in el.iterchildren():
            tag = child.tag.split('}')[-1]

            if tag == 'p':
                yield yield_paragraph(child, parent)

            elif tag == 'tbl':
                yield Table(child, parent)

            elif tag == 'sdt':
                text_elements = child.xpath('.//w:t', namespaces={
                    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'})
                full_text = ''.join([t.text for t in text_elements if t.text])
                if full_text.strip():
                    fake_paragraph = Paragraph(child.xpath('.//w:p')[0], parent) if child.xpath(
                        './/w:p') else Paragraph(child, parent)
                    fake_paragraph._text = full_text
                    yield fake_paragraph

                yield from inner_iter(child, parent)

    return inner_iter(parent.element.body, parent)
entries=[]
for a in files:
    file_path = os.path.join(input_folder, a)
    document=Document(file_path)

    text="Введение "
    for block in iter_block_items(document):
        if isinstance(block, SDTParagraph):
            text+=block.text

        elif isinstance(block, Paragraph):
            text+=block.text
        elif isinstance(block, Table):
            z = [tuple(c.text for c in r.cells) for r in block.rows]
            for b in z:
                text+=str(b)

    Prompt=f" Гост {os.path.splitext(a)[0]}: "

    entr={
            "input": Prompt,
            "output": text
        }
    entries.append(entr)
append_to_json_file(json_path,entries)



# for block in iter_block_items(document):
#
#
#     if isinstance(block,SDTParagraph):
#         print(block.text)
#         if "Нормативные" in block.text :
#             should_break = True
    # elif isinstance(block,Paragraph):
    #     print(block.text)
    # elif isinstance(block,Table):
    #     z = [tuple(c.text for c in r.cells) for r in block.rows]
    #     for b in z:
    #         print(b)
    # if should_break:
    #     break