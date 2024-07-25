import fitz
import re
import json

def get_text_from_novel(book_path): 
    ''' Retrieve the raw text from the novel for further processing'''
    doc = fitz.open(book_path)
    data = "".join(page.get_text() for page in doc[52:6247])
    return data

def para_split(text): 
    ''' Split the raw text into paragraphs. 
        This is done by identifying the first uppercasse character with a preceeding new line.
        Usually a new paragraph starts with after a new line and a capital letter.
        This can also break some new lines which are not paragraphs but it doesnt matter.
        '''
    paras = re.split(r'(\n[A-Z])', text)
    new_matches = []
    for i in range(len(paras)):
        if i+1 <= len(paras) and re.match(r'\n[A-Z]',paras[i]):
            new_matches.append(paras[i] + paras[i+1])
            paras[i+1] = ""

        else:
            new_matches.append(paras[i])
        
    new_paras = [i.replace('\n', '') for i in new_matches if i]

    return new_paras

def create_pre_paras(text, pre_paras_path):
    '''Now that we got pre-processed paragraphs, these can be stored for further processing or as direct embeddings '''
    pre_paras = para_split(text)
    para_export = {"pre_paras" : pre_paras}
    with open('resources/pre_paras.json', 'w') as f:
        json.dump(para_export, f)

def create_paragraphs(pre_paras):
    '''Now we can add 3-4 pre-processed paragraphs together as a single paragraph for embeddings.
        Optionally we can use a chunking method on these paragraphs going further.(Not doing that now)'''
    paragraphs = []
    for i in range(0,len(pre_paras), 4): 
        paragraphs.append("\n".join(pre_paras[i:i+4]))

    return paragraphs