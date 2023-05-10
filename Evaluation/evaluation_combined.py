import subprocess

def install(name):
    subprocess.call(['pip', 'install', name])

import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning)
install ("datasets")
install ("rouge_score")
from datasets import  load_metric, list_metrics
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import nltk
nltk.download('punkt')
import pandas as pd
import csv 

hypothesis = []
references = []

with open('output_combined.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        references.append(row[2].split())
        hypothesis.append(row[3].split())


# ------------------------------------------------------
# Rouge
# ------------------------------------------------------

metric = load_metric("rouge")
score_rouge=metric.compute(predictions=hypothesis, references=references)
print("The Rouge score for the combined model is: "+str(score_rouge["rougeL"].mid[2]*100 ))   #fmeasure
#score_rouge.keys()

# ------------------------------------------------------
# Meteor
# ------------------------------------------------------

metric = load_metric("meteor")
score_meteor=metric.compute(predictions=hypothesis, references=references)
print("The METEOR score for the combined model is: "+str(score_meteor['meteor']*100))

# ------------------------------------------------------
# Bleu 1-4
# ------------------------------------------------------

# Load the CSV file into a Pandas DataFrame
data = pd.read_csv('output_combined.csv')

# Extract the reference and hypothesis columns
references = data['question'].tolist()
hypothesis = data['Generated Question'].tolist()

# Tokenize the reference and hypothesis sentences
references = [nltk.word_tokenize(ref.lower()) for ref in references]
hypothesis = [nltk.word_tokenize(hyp.lower()) for hyp in hypothesis]

# Compute the overall BLEU scores for 1-4 n-grams
scores = [0] * 4
smoothie = SmoothingFunction().method4
for i in range(len(references)):
    for j in range(1, 5):
        score_bleu = sentence_bleu([references[i]], hypothesis[i],
                                   weights=tuple(1.0 / j for _ in range(j)),
                                   smoothing_function=smoothie)
        scores[j-1] += score_bleu

for i in range(4):
    scores[i] /= len(references)

# Print the overall BLEU scores
for i in range(1, 5):
    print(f"The BLEU-{i} score is: {scores[i-1] * 100:.2f}")
    