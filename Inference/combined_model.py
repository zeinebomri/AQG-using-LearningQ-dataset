import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import csv
from transformers import T5ForConditionalGeneration, T5TokenizerFast, BartForConditionalGeneration, BartTokenizerFast

# set up the device to use for inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up the T5 model and tokenizer
t5_model = T5ForConditionalGeneration.from_pretrained('results_t5/checkpoint_T5').to(device)
t5_tokenizer = T5TokenizerFast.from_pretrained('t5-base')

# set up the BART model and tokenizer
bart_model = BartForConditionalGeneration.from_pretrained('BART_results/checkpoint').to(device)
bart_tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')

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
        
        # generate the question using T5
        t5_input_ids = t5_tokenizer.encode("ask question: "+ context + " </s>", return_tensors='pt').to(device)
        t5_output = t5_model.generate(t5_input_ids, max_length=256, num_beams=10, early_stopping=True,length_penalty=1.5,no_repeat_ngram_size=2)
        t5_question = t5_tokenizer.decode(t5_output[0], skip_special_tokens=True)

        # generate the question using BART      
        bart_input_ids = bart_tokenizer.encode(context+" </s>" , return_tensors='pt').to(device)
        bart_output = bart_model.generate(bart_input_ids, max_length=256, num_beams=10, early_stopping=True,length_penalty=1.5,no_repeat_ngram_size=2)
        bart_question = bart_tokenizer.decode(bart_output[0], skip_special_tokens=True)

        # compare the two questions and choose the one with fewer rare words and shorter length
        t5_rare_words = len([word for word in t5_question.split() if t5_tokenizer.get_vocab().get(word) is None])
        bart_rare_words = len([word for word in bart_question.split() if bart_tokenizer.get_vocab().get(word) is None])
        if t5_rare_words < bart_rare_words:
            selected_question = t5_question
        elif bart_rare_words < t5_rare_words:
            selected_question = bart_question
        else:
            if len(t5_question) < len(bart_question):
                selected_question = t5_question
            else:
                selected_question = bart_question
        
        # add the extracted keywords and generated question to the row
        row.append(selected_question)
        # add the updated row to the list of rows
        rows.append(row)
        # log message to track progress
        if i % 10 == 0:
            print(f'Processed {i} rows')

    # write the updated rows to the output CSV file
    output_file="output_combined.csv"
    with open(output_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(rows)


