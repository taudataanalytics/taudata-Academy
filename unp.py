# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:25:43 2019
@author: Taufik Sutanto
taufik@tau-data.id
https://tau-data.id

~~Perjanjian Penggunaan Materi & Codes (PPMC) - License:~~
* Modul Python dan gambar-gambar (images) yang digunakan adalah milik dari berbagai sumber sebagaimana yang telah dicantumkan dalam masing-masing license modul, caption atau watermark.
* Materi & Codes diluar point (1) (i.e. "taudata.py" ini & semua slide ".ipynb)) yang digunakan di pelatihan ini dapat digunakan untuk keperluan akademis dan kegiatan non-komersil lainnya.
* Untuk keperluan diluar point (2), maka dibutuhkan izin tertulis dari Taufik Edy Sutanto (selanjutnya disebut sebagai pengarang).
* Materi & Codes tidak boleh dipublikasikan tanpa izin dari pengarang.
* Materi & codes diberikan "as-is", tanpa warranty. Pengarang tidak bertanggung jawab atas penggunaannya diluar kegiatan resmi yang dilaksanakan pengarang.
* Dengan menggunakan materi dan codes ini berarti pengguna telah menyetujui PPMC ini.
"""
import warnings; warnings.simplefilter('ignore')
import tweepy, re, itertools, os
from html import unescape
from tqdm import tqdm
from unidecode import unidecode
from nltk import sent_tokenize
from textblob import TextBlob
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import spacy
import requests
from urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

nlp_en = spacy.load("en_core_web_sm")
lemma_id = StemmerFactory().create_stemmer()

def connect(con="twitter", key=None):
    if con.lower().strip() == "twitter":
        Ck, Cs, At, As = key
        try:
            auth = tweepy.auth.OAuthHandler(Ck, Cs)
            auth.set_access_token(At, As)
            api = tweepy.API(auth)
            usr_ = api.verify_credentials()
            print('Welcome "{}" you are now connected to twitter server'.format(usr_.name))
            return api
        except:
            print("Connection failed, please check your API keys or connection")

def crawlTwitter(api, qry, N = 30, lan='id', loc=None):
    T = []
    if loc:
        print('Crawling keyword "{}" from "{}"'.format(qry, loc))
        for tweet in tqdm(tweepy.Cursor(api.search_tweets, lang=lan, q=qry, count=100, tweet_mode='extended', geocode=loc).items(N)):
            T.append(tweet._json)
    else:
        print('Crawling keyword "{}"'.format(qry))
        for tweet in tqdm(tweepy.Cursor(api.search_tweets, q=qry, lang=lan, count=100, tweet_mode='extended').items(N)):
            T.append(tweet._json)
    print("Collected {} tweets".format(len(T)))
    return T

def adaAngka(s):
    return any(i.isdigit() for i in s)

def fixTags(t):
    getHashtags = re.compile(r"#(\w+)")
    pisahtags = re.compile(r'[A-Z][^A-Z]*')
    tagS = re.findall(getHashtags, t)
    for tag in tagS:
        if len(tag)>0:
            tg = tag[0].upper()+tag[1:]
            proper_words = []
            if adaAngka(tg):
                tag2 = re.split('(\d+)',tg)
                tag2 = [w for w in tag2 if len(w)>0]
                for w in tag2:
                    try:
                        _ = int(w) # error if w not a number
                        proper_words.append(w)
                    except:
                        w = w[0].upper()+w[1:]
                        proper_words = proper_words+re.findall(pisahtags, w)
            else:
                proper_words = re.findall(pisahtags, tg)
            proper_words = ' '.join(proper_words)
            t = t.replace('#'+tag, proper_words)
    return t

def loadCorpus(file='', sep=':', dictionary = True):
    file = open(file, 'r', encoding="utf-8", errors='replace')
    F = file.readlines()
    file.close()
    if dictionary:
        fix = {}
        for f in F:
            k, v = f.split(sep)
            k, v = k.strip(), v.strip()
            fix[k] = v
    else:
        fix = set( (w.strip() for w in F) )
    return fix

def crawlFiles(dPath,types=None): # dPath ='C:/Temp/', types = 'pdf'
    if types:
        return [dPath+f for f in os.listdir(dPath) if f.endswith('.'+types)]
    else:
        return [dPath+f for f in os.listdir(dPath)]

def LoadDocuments(dPath=None,types=None, file = None): # types = ['pdf','doc','docx','txt','bz2']
    Files, Docs = [], []
    if types:
        for tipe in types:
            Files += crawlFiles(dPath,tipe)
    if file:
        Files = [file]
    if not types and not file: # get all files regardless of their extensions
        Files += crawlFiles(dPath)
    for f in Files:
        if f[-3:].lower() in ['txt', 'dic','py', 'ipynb']:
            try:
                df=open(f,"r",encoding="utf-8", errors='replace')
                Docs.append(df.readlines());df.close()
            except:
                print('error reading{0}'.format(f))
        else:
            print('Unsupported format {0}'.format(f))
    if file:
        Docs = Docs[0]
    return Docs, Files

def LoadStopWords(lang='en'):
    L = lang.lower().strip()
    if L == 'en' or L == 'english' or L == 'inggris':
        from spacy.lang.en import English as lemmatizer
        #lemmatizer = spacy.lang.en.English
        lemmatizer = lemmatizer()
        #lemmatizer = spacy.load('en')
        stops =  set([t.strip() for t in LoadDocuments(file = 'data/stopwords_en.txt')[0]])
    elif L == 'id' or L == 'indonesia' or L=='indonesian':
        from spacy.lang.id import Indonesian
        #lemmatizer = spacy.lang.id.Indonesian
        lemmatizer = Indonesian()
        stops = set([t.strip() for t in LoadDocuments(file = 'data/stopwords_id.txt')[0]])
    else:
        print('Warning, language not recognized. Empty StopWords Given')
        stops = set(); lemmatizer = None
    return stops, lemmatizer

def cleanText(T, fix={}, onlyChar=True, lemma=False, lan='id', stops = set(), symbols_remove = True, min_charLen = 2, max_charLen = 15, fixTag= True):
    pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    t = re.sub(pattern,' ',T) #remove urls if any
    pattern = re.compile(r'ftp[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    t = re.sub(pattern,' ',t) #remove urls if any
    t = unescape(t) # html entities fix
    if fixTag:
        t = fixTags(t) # fix abcDef
    t = t.lower().strip() # lowercase
    t = unidecode(t)
    t = ''.join(''.join(s)[:2] for _, s in itertools.groupby(t)) # remove repetition
    t = t.replace('\n', ' ').replace('\r', ' ')
    t = sent_tokenize(t) # sentence segmentation. String to list
    for i, K in enumerate(t):
        if symbols_remove:
            listKata = re.sub(r'[^.,_a-zA-Z0-9 -\.]',' ',K)

        listKata = TextBlob(listKata).words
        if fix:
            for j, token in enumerate(listKata):
                if str(token) in fix.keys():
                    listKata[j] = fix[str(token)]

        if onlyChar:
            listKata = [tok for tok in listKata if sum([1 for d in tok if d.isdigit()])==0]

        if stops:
            listKata = [tok for tok in listKata if str(tok) not in stops and len(str(tok))>=min_charLen]
        else:
            listKata = [tok for tok in listKata if len(str(tok))>=min_charLen]

        if lemma and lan.lower().strip()=='id':
            t[i] = lemma_id.stem(' '.join(listKata))
        elif lemma and lan.lower().strip()=='en':
            listKata = [str(tok.lemma_) for tok in nlp_en(' '.join(listKata))]
            t[i] = ' '.join(listKata)
        else:
            t[i] = ' '.join(listKata)

    return ' '.join(t) # Return kalimat lagi

def print_Topics(model, feature_names, Top_Topics, n_top_words):
    for topic_idx, topic in enumerate(model.components_[:Top_Topics]):
        print("Topic #%d:" %(topic_idx+1))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))