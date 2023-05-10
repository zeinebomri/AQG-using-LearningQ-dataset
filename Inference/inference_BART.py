import csv
import torch
import os
import pandas as pd
import torch
from transformers import BartForConditionalGeneration, BartTokenizerFast, AdamW
from torch.utils.data import Dataset, DataLoader

# set up the device to use for inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BartForConditionalGeneration.from_pretrained('checkpoint').to(device)
tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')

input_file = 'PreprocessedData/test.csv'

with open(input_file, 'r') as csv_file:
    # initialize the CSV reader
    csv_reader = csv.reader(csv_file)
    # get the header row and add the new columns
    header = next(csv_reader)
    header.append('Generated Question')
    # create a new list to hold the rows with the new columns
    rows = [header]
    # iterate over each row in the CSV file
    for i, row in enumerate(csv_reader):
        # get the context from the current row
        context = row[1]
        # encode the context and keywords using the BART tokenizer
        input_ids = tokenizer.encode(context+" </s>" , return_tensors='pt').to(device)
        # generate the question based on the input context
        output = model.generate(input_ids, max_length=256, num_beams=10, early_stopping=True,length_penalty=1.5,no_repeat_ngram_size=2)
        # decode the generated question using the tokenizer
        question = tokenizer.decode(output[0], skip_special_tokens=True)
        # add the generated question to the row
        row.append(question)
        # add the updated row to the list of rows
        rows.append(row)
        # log message to track progress
        if i % 10 == 0:
            print(f'Processed {i} rows')

    output_file="output_BART.csv"
    # write the updated rows to the output CSV file
    with open(output_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(rows)

