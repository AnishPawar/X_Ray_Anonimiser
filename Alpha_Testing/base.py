from typing import Pattern
import nltk

from nltk.corpus import stopwords

ex = ''' 
ABHISHEK PATEL DR SUSHANT YADAV 2901002020p1 CHEST PA 10/29/2020
AR HOSPITAL MALAD (W). PH:022 28882288
'''

final_wordlist = ex.split(' ')

# final_wordlist =['Status', 'laufende', 'Projekte', 'bei', 'Stand', 'Ende', 'diese', 'Bei']

stopwords_ger = stopwords.words('english')
filtered_words = [w for w in final_wordlist if w.lower() not in stopwords_ger]
print(filtered_words)


# nltk.download()

# from nltk.tokenize import word_tokenize
# from nltk.tag import pos_tag



# ex = ex.lower()

# def preprocess(sent):
#     sent = nltk.word_tokenize(sent)
#     sent = nltk.pos_tag(sent)

#     return sent

# sent = preprocess(ex)
# print(sent)
# # pattern = 'NP: {<DT>?<JJ>*<NN>}'
# pattern = r''' 
#                        NAME: 
#                        {<NNP>+} 
#                        '''
# cp = nltk.RegexpParser(pattern)
# cs = cp.parse(sent)
# # print(cs)

# from nltk.chunk import conlltags2tree, tree2conlltags
# from pprint import pprint
# iob_tagged = tree2conlltags(cs)
# # pprint(iob_tagged)

# ne_tree = nltk.ne_chunk(pos_tag(word_tokenize(ex)))
# # print(ne_tree)













# # import nltk.tokenize as nt
# # import nltk

# # def extract_NN(sent):
# #     grammar = r"""
# #     NBAR:
# #         # Nouns and Adjectives, terminated with Nouns
# #         {<NN.*>*<NN.*>}

# #     NP:
# #         {<NBAR>}
# #         # Above, connected with in/of/etc...
# #         {<NBAR><IN><NBAR>}
# #     """
# #     chunker = nltk.RegexpParser(grammar)
# #     ne = set()
# #     chunk = chunker.parse(nltk.pos_tag(nltk.word_tokenize(sent)))
# #     for tree in chunk.subtrees(filter=lambda t: t.label() == 'NP'):
# #         ne.add(' '.join([child[0] for child in tree.leaves()]))
# #     return ne


# # text=''' 

# # ABHISHEK PATEL DR SUSHANT YADAV 2901002020p1 CHEST PA 10/29/2020
# # AR HOSPITAL MALAD (W). PH:022 28882288
# # '''
# # ss=nt.sent_tokenize(text)
# # tokenized_sent=[nt.word_tokenize(sent) for sent in ss]
# # pos_sentences=[nltk.pos_tag(sent) for sent in tokenized_sent]
# # # print(pos_sentences)

# # ne = extract_NN(text)
# # print(ne)