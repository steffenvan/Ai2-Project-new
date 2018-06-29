import spacy

nlp = spacy.load("en")

doc = nlp("bikes")

for token in doc :
    print(token.text)
    print(token.pos_)