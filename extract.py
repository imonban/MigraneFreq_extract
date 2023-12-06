
import pandas as pd
import spacy
import pandas as pd
from random import shuffle
import re #regexes
import sys #command line arguments
import os, os.path
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize


#nlp = spacy.load("en_core_web_sm")
#nlp = en_core_web_sm.load()
def word_steming(sent):
    stemmer = SnowballStemmer("english")
    #print(outputstr)
    #print exclude_list         
    words = sent.split(' ')
    words = [stemmer.stem(str(word)) for word in words]
    newContent = ' '.join([word for word in words])
    return newContent

def sentence_extraction(rawnote):
    rawnote = rawnote.split('PLAN:')[0]       
    Uterms = ['headache', 'migraine', 'head ache', 'headaches', 'head aches', 'frequency']
    feq_des= ['daily', 'days','days per month', 'days per mon','intensity', 'freqency', 'hours per month', 'hours per mon','/month']
    #content  = rawnote.lower();
    str1 = rawnote
    str1 = ' '+str1+' '
    sentences = []
    str1 = re.sub(r'[^\x00-\x7f]',r' ', str1)
    re1='(\\d+)'	# Integer Number 1
    re2='(\\.)'	# Any Single Character 1
    re3='( )'	# White Space 1
    re4='((?:[a-z][a-z]+))'	# Word 1

    rg = re.compile(re1+re2+re3+re4,re.IGNORECASE|re.DOTALL)
    str1 = re.sub(rg,r'\n', str1)
    re1='((?:[a-z][a-z]+))'	# Word 1
    re2='(:)'	# Any Single Character 1
    re3='(\\s+)'	# White Space 1
    re4 = ';'
    rg = re.compile(re1+re2+re3,re.IGNORECASE|re.DOTALL)
    m = rg.search(str1)
    if m:
        str1 = re.sub(rg,'\n'+m.group(1)+':', str1)
    str1 = re.sub('\s\s\s+','\n',str1)
    str1 = str1.replace('\r', '\n')
    str1 = str1.replace('--', '\n')
    str1 = str1.replace('#', '\n')
    str1 = str1.replace('?', '\n')
    str1 = str1.replace(';', '\n')
    str1 = str1.replace(':', '\n')
    paragraphs = [p for p in str1.split('\n') if p]
    #paragraphs = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', str1)
    #paragraphs = filter(None, re.split("([A-Z][^A-Z]*)", str1))
    for paragraph in paragraphs:
        temp_sentences = sent_tokenize(paragraph)
        for term in Uterms:
            for i in range(len(temp_sentences)):
                temp_sentences[i] = temp_sentences[i].lower()
                #if term in temp_sentences[i]: ## Alternative approach: more flexible
                if re.search(r'\b' + term + r'\b', temp_sentences[i]):
                    if any(word in temp_sentences[i] for word in feq_des):
                        sentences.append(temp_sentences[i])
    
    return sentences
def sentence_extraction_MHS(txt):
#def sentence_extraction(txt):
    
    try:
        rawnote = str(txt)
    except:
        rawnote = ' '
    if 'Summary of headache:' in txt:
        txt = txt.split('Summary of headache:')[1].split('MEDICAL HISTORY')[0].split('Prior medication trials:')[0]
        return txt.lower().replace('\n', ' ')
    #rawnote = txt
    #rawnote = rawnote.split('PLAN:')[0]       
    Uterms = ['headache', 'migraine', 'head ache', 'headaches', 'head aches']
    feq_des= ['days per month', 'days per mon','intensity', 'freqency', 'hours per month', 'hours per mon','daily', 'times','time','per month', '/month', 'per week', '/week']
    #content  = rawnote.lower();
    str1 = rawnote
    str1 = ' '+str1+' '
    sentences = []
    str1 = re.sub(r'[^\x00-\x7f]',r' ', str1)
    re1='(\\d+)'	# Integer Number 1
    re2='(\\.)'	# Any Single Character 1
    re3='( )'	# White Space 1
    re4='((?:[a-z][a-z]+))'	# Word 1

    rg = re.compile(re1+re2+re3+re4,re.IGNORECASE|re.DOTALL)
    str1 = re.sub(rg,r'\n', str1)
    re1='((?:[a-z][a-z]+))'	# Word 1
    re2='(:)'	# Any Single Character 1
    re3='(\\s+)'	# White Space 1
    re4 = ';'
    rg = re.compile(re1+re2+re3,re.IGNORECASE|re.DOTALL)
    m = rg.search(str1)
    if m:
        str1 = re.sub(rg,'\n'+m.group(1)+':', str1)
    str1 = re.sub('\s\s\s+','\n',str1)
    str1 = str1.replace('\r', '\n')
    str1 = str1.replace('\t', '\n')
    str1 = str1.replace('--', '\n')
    str1 = str1.replace('#', '\n')
    str1 = str1.replace('?', '\n')
    str1 = str1.replace(';', '\n')
    str1 = str1.replace(':', '\n')
    paragraphs = [p for p in str1.split('\n\n') if p]
    #paragraphs = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', str1)
    #paragraphs = filter(None, re.split("([A-Z][^A-Z]*)", str1))
    for paragraph in paragraphs:
        paragraph = paragraph.replace('\n', ' ')
        temp_sentences = sent_tokenize(paragraph)
        for term in Uterms:
            for i in range(len(temp_sentences)):
                temp_sentences[i] = temp_sentences[i].lower()
                #if term in temp_sentences[i]: ## Alternative approach: more flexible
                if re.search(r'\b' + term + r'\b', temp_sentences[i]):
                    #print(temp_sentences[i])
                    if any(word in temp_sentences[i] for word in feq_des):
                        sentences.append(temp_sentences[i])
    return sentences
