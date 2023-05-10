# ------------------------------------------------------
# Import Required Modules
# ------------------------------------------------------
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import pandas as pd
import subprocess
import itertools

def install(name):
    subprocess.call(['pip', 'install', name])
install ("transformers")
install ("datasets")
install ("sentencepiece")

from transformers import (BartTokenizerFast,BartForConditionalGeneration,Trainer,TrainingArguments,EarlyStoppingCallback)
import datasets
from datasets import load_dataset, load_metric, list_metrics, Dataset

# ------------------------------------------------------
# Prepare inputs
# ------------------------------------------------------

#train data
train = pd.read_csv("PreprocessedData/train.csv", usecols = ['context','question'],engine='python',error_bad_lines=False)
train=train.to_dict(orient="list")

#valid data
valid = pd.read_csv("PreprocessedData/valid.csv", usecols = ['context','question'],engine='python',error_bad_lines=False)
valid=valid.to_dict(orient="list")

dd = datasets.DatasetDict({"train":Dataset.from_dict(train),"valid":Dataset.from_dict(valid)})
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = "facebook/bart-base"
model = BartForConditionalGeneration.from_pretrained(checkpoint).to(device)
tokenizer = BartTokenizerFast.from_pretrained(checkpoint)
tokenizer.sep_token = '<sep>'
tokenizer.add_tokens(['<sep>'])


max_input_length =  512
max_target_length = 64

# tokenize the examples
def convert_to_features(example_batch):

    input_encodings = tokenizer.batch_encode_plus(example_batch['context'], 
                                                  max_length=max_input_length, 
                                                  add_special_tokens=True,
                                                  truncation=True, 
                                                  pad_to_max_length=True)
    
    target_encodings = tokenizer.batch_encode_plus(example_batch['question'], 
                                                   max_length=max_target_length, 
                                                   add_special_tokens=True,
                                                   truncation=True, pad_to_max_length=True)
                                                   
    encodings = {
        'input_ids': input_encodings['input_ids'], 
        'attention_mask': input_encodings['attention_mask'],
        'decoder_input_ids': target_encodings['input_ids']
        ,'decoder_attention_mask': target_encodings['attention_mask']
    }

    return encodings

def add_eos_examples(example):
  example['context'] = example['context'] + " </s>"
  example['question'] = example['question'] + " </s>"
  return example

def add_special_tokens(example):
  example['question'] = example['question'].replace("{sep_token}", '<sep>')
  return example
  
tokenized_dataset  = dd.map(add_eos_examples)
tokenized_dataset = tokenized_dataset.map(add_special_tokens)
tokenized_dataset  = tokenized_dataset.map(convert_to_features,  batched=True)

tokenized_dataset = tokenized_dataset.remove_columns(["context", "question"])
train_dataset = tokenized_dataset["train"]
valid_dataset = tokenized_dataset["valid"]
columns = ['input_ids', 'decoder_input_ids', 'attention_mask', 'decoder_attention_mask']
train_dataset.set_format(type='torch', columns=columns)
valid_dataset.set_format(type='torch', columns=columns)

torch.save(train_dataset, './BART_results/train_data.pt')
torch.save(valid_dataset, './BART_results/valid_data.pt')



#collate a list of Dataset samples into a batch, and return a dictionary of tensors
@dataclass
class T2TDataCollator():
  def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([example['input_ids'] for example in batch])
    lm_labels = torch.stack([example['decoder_input_ids'] for example in batch])
    lm_labels[lm_labels[:, :] == 0] = -100 #mask the padding tokens in the input sequences to ignore them during training
    attention_mask = torch.stack([example['attention_mask'] for example in batch])
    decoder_attention_mask = torch.stack([example['decoder_attention_mask'] for example in batch])
    
    return {
        'input_ids': input_ids, 
        'attention_mask': attention_mask,
        'labels': lm_labels, 
        'decoder_attention_mask': decoder_attention_mask
    }

# ------------------------------------------------------
# Training loop
# ------------------------------------------------------

# Define the hyperparameters to tune
hyperparameters = {
    'learning_rate': [1e-4, 5e-5],
    'weight_decay': [0.01, 0.001]
}

# Initialize variables to keep track of the best hyperparameters and validation loss
best_hyperparameters = None
best_valid_loss = float('inf')

# Generate all combinations of hyperparameters
param_combinations = itertools.product(*hyperparameters.values())


# Loop over all hyperparameter combinations
for params in param_combinations:
    learning_rate, weight_decay = params
    
    # Define the TrainingArguments
    training_args = TrainingArguments(
        output_dir="./BART_results/run",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=16,
        learning_rate=learning_rate,
        num_train_epochs=10,
        logging_steps=100,
        evaluation_strategy="steps",
        save_steps=500,
        eval_steps=500,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        save_total_limit=3
    )


    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=T2TDataCollator(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train the model
    trainer.train()

    # Save the model checkpoint
    valid_loss = trainer.evaluate().get('eval_loss')
    if valid_loss < best_valid_loss:
        best_hyperparameters = params
        best_valid_loss = valid_loss
        model.save_pretrained('checkpoint')


# Print the best hyperparameters and validation loss
print(f"Best hyperparameters: {best_hyperparameters}")
print(f"Best validation loss: {best_valid_loss}")