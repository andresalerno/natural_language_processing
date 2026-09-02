# para execução: python .\python\introduction\intro_pt.py
import nltk
import spacy                            # biblioteca moderna de PLN, muito usada para tokenização, POS tagging, entidades etc.
import numpy as np
import pandas as pd
import copy as cp
import joblib
from pathlib import Path

sentence = "Em 2022 houve uma acirrada eleição entre Bolsonaro e Lula."
# Aqui você está fazendo uma tokenização extremamente simples. O Python divide a frase nos espaços.
words = sentence.split()
bag_of_words = cp.deepcopy(words)
np.random.shuffle(bag_of_words)
# Bag of words:
print(bag_of_words)

# Annotated words:
# nltk.download('popular')
# Using natural language toolkit
print("Usando o natural language toolkit:")
# Use lang with ISO 639 code of the language
# POS: Part-of-Speech
#pos_tags = nltk.pos_tag(sentence.split(), lang="pt") # not implemented.

# Reference to get the trained model:
# https://github.com/inoueMashuu/POS-tagger-portuguese-nltk
trained_data_folder = Path(__file__).parent / 'data'
portuguese_tagger = joblib.load(trained_data_folder / 'POS_tagger_brill.pkl')
pos_tags = portuguese_tagger.tag(nltk.wordpunct_tokenize(sentence))
print(pos_tags)
pos_tags_df = pd.DataFrame(pos_tags).T
print(pos_tags_df)

print("Usando o Spacy para se obter as partes do discurso.")
## https://spacy.io/models/pt
model_spacy = spacy.load('pt_core_news_sm')
pos_tags_2 = [ (word, word.pos_) for word in model_spacy(sentence)]
pos_tags_2_df = pd.DataFrame(pos_tags_2).T
print(pos_tags_2)
print(pos_tags_2_df)



