import subprocess

def install(name):
    subprocess.call(['pip', 'install', name])
    
install ("gensim")
install ("summa")

import csv
import torch
from transformers import BartForConditionalGeneration, BartTokenizerFast
import gensim
from summa import keywords

# set up the device to use for inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BartForConditionalGeneration.from_pretrained('checkpoint_keywords_BART').to(device)
tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')

input_file = 'PreprocessedData/test.csv'

def extract_keywords(context):
    # Extract the keywords using the TextRank algorithm
    keyword_str = keywords.keywords(context,ratio=0.2).replace('\n', ' ') #ratio is set to extract the 20% of the most significant words from the text, you can adjust it
    return keyword_str

with open(input_file, 'r') as csv_file:
    # initialize the CSV reader
    csv_reader = csv.reader(csv_file)
    # get the header row and add the new columns
    header = next(csv_reader)
    header.append('Keywords')
    header.append('Generated Question')
    # create a new list to hold the rows with the new columns
    rows = [header]

    # iterate over each row in the CSV file
    for i, row in enumerate(csv_reader):
        # get the context from the current row
        context = row[1]
        print("context", context)
        # extract the keywords from the context
        keywords_extracted = extract_keywords(context)
        # encode the context and keywords using the BART tokenizer
        input_ids = tokenizer.encode(context + '<sep>' + keywords_extracted+ " </s>", return_tensors='pt').to(device)
        # generate the question based on the input context
        output = model.generate(input_ids, max_length=256, num_beams=10, early_stopping=True,length_penalty=1.5,no_repeat_ngram_size=2)
        # decode the generated question using the tokenizer
        question = tokenizer.decode(output[0], skip_special_tokens=True)
        # add the extracted keywords and generated question to the row
        row.append(keywords_extracted)
        row.append(question)
        # add the updated row to the list of rows
        rows.append(row)
        # log message to track progress
        if i % 10 == 0:
            print(f'Processed {i} rows')
    
    output_file="output_BART_keywords.csv"
    # write the updated rows to the output CSV file
    with open(output_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(rows)