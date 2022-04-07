# Basic libraries needed
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive

# Import SpaCy
import spacy
from spacy import displacy

# Import libraries needed for summarization
from string import punctuation
from heapq import nlargest

# Import NLTK
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

from collections import Counter


## PREPROCESSING ##

# Load and drop unnecessary rows
df = pd.read_csv('dont_say_gay.csv')
df = df[df['text'].str.match(r'^F L O *') == False]
df = df.iloc[:, 1]

# Convert from dataframe to string
bill_text = df.values.tolist()
bill_text = ' '.join(bill_text)
bill_text

# Get rid of the junk
def clean_text(string):
  string = string.lower()
  string = re.sub(r'[\d:,\.\(\)]', '', string)
  string = re.sub(r'\s+page\s+of', '', string)
  string = re.sub(r'words\s+(underlined|stricken)', '', string)
  string = re.sub(r'(hb--er)|(are\s+deletions|additions)', '', string)
  return string

cleaned = clean_text(bill_text)


## TEXT SUMMARIZATION ##

def summarize(text, per):
    nlp = spacy.load('en_core_web_sm')
    stop = stopwords.words('english')
    doc = nlp(text)
    tokens=[token for token in doc]
    word_frequencies={}
    for word in doc:
        if word.text.lower() not in list(stop):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    max_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/max_frequency
    sentence_tokens= [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():                            
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
    select_length=int(len(sentence_tokens)*per)
    summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
    final_summary=[word.text for word in summary]
    summary=''.join(final_summary)
    return summary

summarize(cleaned, 0.06)

# Get rid of semicolons for rest of analysis
cleaned = re.sub(r';', '', cleaned)


## NAMED ENTITY RECOGNITION ##

ner = spacy.load("en_core_web_sm")
doc = ner(cleaned)

displacy.render(doc, style="ent", jupyter=True)


## UNIGRAM ANALYSIS ##

stop = stopwords.words('english')

tokens = word_tokenize(bill_text)

no_stops = [word for word in tokens if word not in stop]

pos = nltk.pos_tag(no_stops)

lemmatizer = WordNetLemmatizer()

wordnet_dict = {
    "NN": wordnet.NOUN,
    "VB": wordnet.VERB,
    "JJ": wordnet.ADJ,
    "RB": wordnet.ADV
}

word_lemmas = []
for seq in pos:
  if seq[1][:2] in wordnet_dict.keys():
    word_lemmas.append(lemmatizer.lemmatize(seq[0], pos = wordnet_dict.get(seq[1][:2])))


count_words = Counter(word_lemmas)

top_words = count_words.most_common(10)

plt.figure(figsize=(10,6))
ax = sns.barplot(
    x=[word[0] for word in top_words],
    y=[count[1] for count in top_words],
    palette="mako"
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)


## BIGRAM ANALYSIS ##

two_words = nltk.bigrams(word_lemmas)

frequency = nltk.FreqDist(two_words)

most_two_words = frequency.most_common(10)

strings = []
for word in most_two_words:
  string = "_".join(word[0])
  strings.append(string)
  string = ""
print(strings)

plt.figure(figsize=(10,6))
ax = sns.barplot(
    x=[most_two_words[count][1] for count in range(len(most_two_words))],
    y=strings,
    palette="mako"
)
