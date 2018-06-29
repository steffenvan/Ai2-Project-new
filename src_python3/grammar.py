import spacy
from spacy.symbols import nsubj, VERB
from spacy import displacy

nlp = spacy.load('en_core_web_sm')

doc = nlp("The cat and the dog sleep in the basket near the door like cow")
# print (doc)
# prev_token = None
# for token in doc:
#     prev_token = token
#     print(prev_token)
    # if prev_token == 'DET' and token == 'NOUN':
    #     print(prev_token, token)
#
det = []
det_obj = []
noun = []
index = 0

for token, token_next in zip(doc, doc[1:]):
    if token.pos_ == 'DET':
        word = token.text + ' ' + token_next.text
        det.append(word)
    elif token.pos_ == 'NOUN':
        det_obj.append(token)

for token in doc:
    if token.pos_ == 'NOUN' and token not in det_obj:
        noun.append(token.text)

print(det)
print(noun)
