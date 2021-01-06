import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import pytesseract 
import cv2
from pytesseract import Output


nlp = en_core_web_sm.load()

# base = cv2.imread("X_Ray_Test_2.jpg")

# text = pytesseract.image_to_string(base,lang='eng')
# print(text)

# cv2.imshow('image',base)
# cv2.waitKey(0)

text = ''' 

VIRAJ THAKKAR 2013Apri9 21:06 MB VIRAJ THAKKAR W1IApri9 21:07
tea is gt = emP Gen
ig3e 6 MB

   

3.8 4?

F |

Home | Home/set ! Label. | Symbols, | xX Word! Done Home ' Homerser' Laver | Symons " X word’ Oone

VIRAJ THAKKAR 2013Apri9 21:07 BB VIRAJ THAKKAR
— . _ = SmP Color 9
L3Q 1244Hz
= 9

8

+ 68% 4
a :
os °°
Tis

, 02

QO

O
47 Ocne

  

| VIRAJS THAKKAR 2013Apri9 21:08 Mi VIRAJ THAKKAR 2013Aprig 21-03
uss S MB
=
ug
* 68%
Mi
0.8
TIs
. 0.2
utp

ak

ar

 
'''

doc = nlp(text)

print(doc)

# print([(X.text, X.label_) for X in doc.ents]) 