import pandas as pd
import csv

dataInti = pd.read_csv(r'data/dataclean.csv')
dataInti = dataInti[['username', 'full_text', 'cleaned_text']]
dataInti = dataInti.dropna()

inPos = dict()
with open('kamus/inset/positive.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        inPos[row[0]] = int(row[1])

inNeg = dict()
with open('kamus/inset/negative.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        inNeg[row[0]] = int(row[1])

def inset(text):
    score = 0
    text = text.replace("'", "").replace("[", "").replace("]", "").split(', ')
    for word in text:
        if (word in inPos):
            score = score + inPos[word]
    for word in text:
        if (word in inNeg):
            score = score + inNeg[word]
    polarity = ''
    if (score > 0):
        polarity = "Positif"
    elif (score < 0):
        polarity = "Negatif"
    else:
        polarity = "Netral"
    return score, polarity

labelInset = dataInti['cleaned_text'].apply(inset)
labelInset = list(zip(*labelInset))
dataInti['skorKlasifikasi'] = labelInset[0]
dataInti['klasifikasi'] = labelInset[1]
print(dataInti['klasifikasi'].value_counts()),dataInti.shape

dataInti['label'] = dataInti['klasifikasi'].map({'Negatif': -1,
                                                 'Netral': 0,
                                                 'Positif': 1})

dataInti.to_csv("data/label.csv", index = False, header = True, index_label=None)