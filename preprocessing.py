import re
import json
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory as stopFactory,
    StopWordRemover as stopRemove,
    ArrayDictionary as arrayDic)
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory as stemfac

with open("kamus/kamusAlay/_json_kamus.txt") as kamus:
    slang = kamus.read()
kamusSlang = json.loads(slang)

dataInti = pd.read_csv("data/data(2000).csv")
dataInti = dataInti[['username', 'full_text']]
dataInti = dataInti.dropna()

#Data Cleansing
def cleanData(text):
    #penghapusan non-ascii
    text = re.sub(r'[^\x00-\x7F]+',' ', text)
    #penghapusan url1
    text = re.sub(r'http\S+',' ', text)
    #penghapusan url2
    text = re.sub(r'pic.twitter.com\S+',' ', text)
    #penghapusan mention
    text = re.sub(r'\@([\w]+)',' ', text)
    #penghapusan hashtag
    text = re.sub(r'\#([\w]+)',' ', text)
    #penghapusan simbol
    text = re.sub(r'[!$%^&*@#()_+|~=`{}\[\]%\-:";\'<>?,.\/]', ' ', text)
    #koreksi huruf berlebih
    text = re.sub(r'([a-zA-Z])\1\1','\\1', text)
    #penghapusan angka
    text = re.sub(r'[0-9]+', '', text)
    #penghapusan spasi ganda
    text = re.sub(' +', ' ', text)
    #penghapusan spasi di awal kalimat
    text = re.sub(r'^[ ]|[ ]$','', text)
    return text

#Case Folding
def caseFold(text):
    text = text.lower() # Mengecilkan semua huruf
    return text

#SlangRemoval
def slangremove(text):
    #pembersihan kata slang dari dataset menggunakan bantuan kamus alay
    text = " ".join(kamusSlang.get(ele, ele) for ele in text.split())
    return text

#Stopword
stopTambah = ["yuhuu", "yuhu"]
stopWord = stopFactory().get_stop_words()
stopWord.extend(stopTambah)

arrayBaru = arrayDic(stopWord)
stopWordRemover = stopRemove(arrayBaru)
def stopwords(text):
    # membersihkan stopword dari dataset seperti di, dan, akan
    text = stopWordRemover.remove(text)
    return text

#Stemming
def stemming(text):
    fac = stemfac()
    stemmer = fac.create_stemmer()
    text = [stemmer.stem(word) for word in text]
    return text

#Preprocess data
dataInti['cleaned_text'] = dataInti['full_text'].apply(cleanData)
dataInti['cleaned_text'] = dataInti['cleaned_text'].apply(caseFold)
dataInti['cleaned_text'] = dataInti['cleaned_text'].apply(slangremove)
dataInti['cleaned_text'] = dataInti['cleaned_text'].apply(lambda x:stopwords(x))
dataInti['cleaned_text'] = dataInti['cleaned_text'].apply(lambda x:x.split())
dataInti['cleaned_text'] = dataInti['cleaned_text'].apply(stemming)

dataInti.drop_duplicates(subset=['cleaned_text'], keep = 'first', inplace = True)

dataInti.to_csv(r'data/dataclean.csv', index = False, header = True, index_label=None)