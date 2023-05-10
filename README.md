# AQG-using-LearningQ-dataset
Automatic question generation using Transfer Learning Techniques



Three different approaches are suggested to address the Automatic Question Generation problem using the LearningQ dataset introduced by in this [paper](https://ojs.aaai.org/index.php/ICWSM/article/view/14987/14837).
Essentially, we fine-tune the pretrained transformer models, namely BART and T5, for the downstream task of question generation.

********
     -First approach: the text is used as the sole input to the model.
     -Second approach: we start by extracting the keywords from the text and subsequently use both the
         text and the extracted keywords as inputs to fine-tune the transformers. 
     -Third approach: we also fine-tune the transformers, and then combine their predictions into an ensemble learning model. 
********

The experiments have shown that while all the approaches we proposed resulted in improved METEOR scores compared to previous findings from similar
studies, the combined model produced the most favorable results and led to the highest overall improvement in the METEOR score. Nevertheless, it should 
be noted that while the keywords approach can be useful in some aspects, it might not be effective in helping the models generate
questions close to the ground truth. 

The work presented experiments with the aforementioned approaches, aiming to contribute to the development of question generation systems.
