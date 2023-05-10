from flask import Flask, render_template, request
from transformers import BartForConditionalGeneration, BartTokenizerFast
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import torch
from summa import keywords

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_keywords(context):
    # Extract the keywords using the TextRank algorithm
    keyword_str = keywords.keywords(context,ratio=0.2).replace('\n', ' ') #ratio is set to extract the 20% of the most significant words from the text, you can adjust it
    return keyword_str

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index_v4.html')

@app.route('/', methods=['POST'])
def generate_question():
    context = request.form['context']
    n = request.form['n']
    if (context==""):
        return render_template('index_v4.html', question="Please make sure to enter your text in the box!",n=n)
    if (n==""):
        return render_template('index_v4.html', question="Please make sure to enter the number of questions!",context=context)
    
    n = int(request.form['n'])
    try:
        model_name = request.form['model']
        if model_name=="BART":
            # Load the model
            model = BartForConditionalGeneration.from_pretrained("models/BART").to(device)
            tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
            input_ids = tokenizer.encode(context + " </s>" , return_tensors='pt').to(device)
            questions = []
             
            i = 0
            while i < n:
                output = model.generate(input_ids, max_length=256, num_beams=10, early_stopping=True, length_penalty=1, no_repeat_ngram_size=2, num_return_sequences=n)
                for seq in output:
            # decode the generated question using the tokenizer
                    question = tokenizer.decode(seq, skip_special_tokens=True)
                    if question not in questions:
                       questions.append(question)
                       i += 1
                    if i >= n:
                       break


        elif model_name == "T5":
            model = T5ForConditionalGeneration.from_pretrained("models/T5").to(device)
            tokenizer = T5TokenizerFast.from_pretrained('t5-base')
            input_ids = tokenizer.encode("ask question: "+ context + " </s>", return_tensors='pt').to(device)
            questions = []

            i = 0
            while i < n:
                output = model.generate(input_ids, max_length=256, num_beams=10, early_stopping=True, length_penalty=1, no_repeat_ngram_size=2, num_return_sequences=n)
                for seq in output:
            # decode the generated question using the tokenizer
                    question = tokenizer.decode(seq, skip_special_tokens=True)
                    if question not in questions:
                       questions.append(question)
                       i += 1
                    if i >= n:
                       break

        elif model_name=="BART with keywords":
            model3 = BartForConditionalGeneration.from_pretrained("models/BART with keywords").to(device)
            model = model3
            tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
            keywords_extracted = extract_keywords(context)
            input_ids = tokenizer.encode(context +" " + keywords_extracted+ " </s>", return_tensors='pt').to(device)
            questions = []

            i = 0
            while i < n:
                output = model.generate(input_ids, max_length=256, num_beams=10, early_stopping=True, length_penalty=1, no_repeat_ngram_size=2, num_return_sequences=n)
                for seq in output:
            # decode the generated question using the tokenizer
                    question = tokenizer.decode(seq, skip_special_tokens=True)
                    if question not in questions:
                       questions.append(question)
                       i += 1
                    if i >= n:
                       break

        elif  model_name=="T5 with keywords" :
            model4 = T5ForConditionalGeneration.from_pretrained("models/T5 with keywords").to(device)

            model = model4
            tokenizer = T5TokenizerFast.from_pretrained('t5-base')
            keywords_extracted = extract_keywords(context)
            input_ids = tokenizer.encode("ask question: "+context + '<sep>' + keywords_extracted+ " </s>", return_tensors='pt').to(device)
            questions = []

            i = 0
            while i < n:
                output = model.generate(input_ids, max_length=256, num_beams=10, early_stopping=True, length_penalty=1, no_repeat_ngram_size=2, num_return_sequences=n)
                for seq in output:
            # decode the generated question using the tokenizer
                    question = tokenizer.decode(seq, skip_special_tokens=True)
                    if question not in questions:
                       questions.append(question)
                       i += 1
                    if i >= n:
                       break
        else:
            bart_model = BartForConditionalGeneration.from_pretrained("models/BART").to(device)
            bart_tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
            
            t5_model = T5ForConditionalGeneration.from_pretrained("models/T5").to(device)
            t5_tokenizer = T5TokenizerFast.from_pretrained('t5-base')


            # generate the question based on the input context using T5
            t5_input_ids = t5_tokenizer.encode("ask question: "+context+ " </s>" , return_tensors='pt').to(device)
            t5_questions = []

            i = 0
            while i < n:
                t5_output = t5_model.generate(t5_input_ids, max_length=256, num_beams=10, early_stopping=True, length_penalty=1, no_repeat_ngram_size=2, num_return_sequences=n)
                for seq in t5_output:
            # decode the generated question using the tokenizer
                    t5_question = t5_tokenizer.decode(seq, skip_special_tokens=True)
                    if t5_question not in t5_questions:
                       t5_questions.append(t5_question)
                       i += 1
                    if i >= n:
                       break



            # generate the question based on the input context using BART
            bart_input_ids = bart_tokenizer.encode(context+ " </s>" , return_tensors='pt').to(device)
            bart_questions = []

            i = 0
            while i < n:
                bart_output = bart_model.generate(bart_input_ids, max_length=256, num_beams=10, early_stopping=True, length_penalty=1, no_repeat_ngram_size=2, num_return_sequences=n)
                for seq in bart_output:
            # decode the generated question using the tokenizer
                    bart_question = bart_tokenizer.decode(seq, skip_special_tokens=True)
                    if bart_question not in bart_questions:
                       bart_questions.append(bart_question)
                       i += 1
                    if i >= n:
                       break            
            
            # compare the two questions and choose the one with fewer rare words and shorter length
            questions=[]
            for i in range(n):
                t5_rare_words = len([word for word in t5_questions[i].split() if t5_tokenizer.get_vocab().get(word) is None])
                bart_rare_words = len([word for word in bart_questions[i].split() if bart_tokenizer.get_vocab().get(word) is None])
                if t5_rare_words < bart_rare_words:
                    questions.append(t5_questions[i])
                elif bart_rare_words < t5_rare_words:
                    questions.append(bart_questions[i])
                else:
                    if len(t5_questions[i]) < len(bart_questions[i]):
                        questions.append(t5_questions[i])
                    else:
                        questions.append(bart_questions[i])
    except Exception as e:
        print(e)
        return render_template('index_v4.html', context=context,question="The text provided couldnt be handled by the model, please rephrase your text or select another model!",n=n, model=model_name)

    return render_template('index_v4.html', context=context, question=questions,n=n, model=model_name)

if __name__ == '__main__':
    app.run(debug=False)

