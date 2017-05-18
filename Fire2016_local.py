# coding=utf-8
# !/usr/bin/python
'''
INFO: This code was used for tweet classification work of FIRE2016 dataset.
DESC:
script options
--------------
--param : parameter list

Created by Samujjwal_Ghosh on 11-Apr-17.

__author__ : Samujjwal Ghosh
__version__ = ": 1 $"
__date__ = "$"
__copyright__ = "Copyright (c) 2017 Samujjwal Ghosh"
__license__ = "Python"

Supervised approaches:
    Naive Bayes,
    SVM,
    Decision Tree,
    Random forests,
    Gaussian Process,
    Adaboost
    
Features:
    # 1. Unigrams, bigrams
    # 2. count of words like (lakh,lakhs,millions,thousands)
    # 3. count of units present (litre,kg,gram)
    # 4. k similar tweets class votes
    # 5. k closest same class distance avg
    # 6. count of frequent words of that class (unique to that class)
    # 7. Length related features.
'''
import os,sys,re,math,json,string,logging
import unicodedata
import heapq
import numpy as np
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn import cluster
from textblob import TextBlob as tb

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
log_file='results.log'
handler=logging.FileHandler(log_file)
handler.setLevel(logging.INFO)

# File Handling-------------------------------------------------------------------------------------
from collections import OrderedDict
from collections import defaultdict
def save_json(dict,filename):
    print("Method: save_json(dict,file)")
    try:
        with open(filename + ".json",'w') as outfile:
            outfile.write(json.dumps(dict,indent=4,sort_keys=True))
        outfile.close()
        return True
    except Exception as e:
        print("Could not write to file: ",filename)
        print("Failure reason: ",e)
        return False

def read_json(filename):
    if os.path.isfile(filename+".json"):
        with open(filename+".json","r",encoding="utf-8") as file:
            json_dict=OrderedDict(json.load(file))
        file.close()
        return json_dict
    else:
        print(file,"** file does not exists,reading [labelled_tweets]")
        lab_tweets=read_json("labelled_tweets")
        # TODO: change from dict to list before passing
        train,test=train_test_split(lab_tweets,test_size=0.3)
        return train

# Globals-------------------------------------------------------------------------------------------
n_classes          =7     # number of classes
n_clusters         =10    # number of clusters
k_clusters_tweet   =3     # hyper-param,# of clusters to assign each tweet (3=nearest 3 clusters)
k_class_tweet      =3     # hyper-param,# of classes to assign each tweet (2=nearest 2 class)
k_similar          =15    # hyper-param,# of similar tweets to find based on cosine similarity
k_unique_words     =25    # hyper-param,# of unique words to find using tf-idf per class
acronym_dict       =read_json("acronym")    # dict to hold acronyms
class_names=['RESOURCES AVAILABLE',
             'RESOURCES REQUIRED',
             'MEDICAL RESOURCES AVAILABLE',
             'MEDICAL RESOURCES REQUIRED',
             'REQUIREMENTS, AVAILABILITY OF RESOURCES AT SPECIFIC LOCATIONS',
             'ACTIVITIES OF VARIOUS NGOs, GOVERNMENT ORGANIZATIONS',
             'INFRASTRUCTURE DAMAGE AND RESTORATION REPORTED'
            ]

# Preprocess----------------------------------------------------------------------------------------
emoticons_str=r'''
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )'''

regex_str=[
    emoticons_str,
    r'<[^>]+>',# HTML tags
    r'(?:@[\w_]+)',# @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",# hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',# URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',# numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",# words with - and '
    r'(?:[\w_]+)',# other words
    r'(?:\S)' # anything else
]

tokens_re  =re.compile(r'('+'|'.join(regex_str)+')',re.VERBOSE | re.IGNORECASE)
emoticon_re=re.compile(r'^'+emoticons_str+'$',re.VERBOSE | re.IGNORECASE)

def preprocess(s,lowercase=False):
    # print("Method: preprocess(s,lowercase=False)")
    tokens=tokens_re.findall(str(s))
    if lowercase:
        tokens=[token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def parse_tweet(tweet):
    # print("Method: parse_tweet(tweet)")
    stop=stopwords.words('english') + list(string.punctuation) + ['rt','via','& amp']
    tweet=re.sub(r"http\S+","urlurl",tweet) # replaces hyperlink with urlurl
    terms=preprocess(tweet,True)
    for term_pos in range(len(terms)):
        terms[term_pos]=terms[term_pos].replace("@","")
        terms[term_pos]=terms[term_pos].replace("#","")
        terms[term_pos]=get_acronyms(terms[term_pos])
        terms[term_pos]=contains_phone(terms[term_pos])
    mod_tweet=" ".join([term for term in terms if term not in stop])
    return mod_tweet

def get_acronyms(term):
    '''Check for Acronyms and returns the acronym of the term'''
    # print("Method: get_acronyms(term)",term)
    global acronym_dict
    if term in acronym_dict.keys():
        # print(term," -> ",acronym_dict[term])
        return acronym_dict[term]
    else:
        return term

# Features------------------------------------------------------------------------------------------
def k_similar_tweets(train,test,k_similar):
    '''Finds k_similar string to the given string by cosine similarity'''
    print("Method: k_similar_tweets(train,new_tweet,k_similar)")
    k_sim_twts=OrderedDict()
    for t_twt_id,t_twt_val in test.items():
        i=0
        sim_list=OrderedDict()
        for tr_twt_id,tr_twt_val in train.items():
            if t_twt_id == tr_twt_id:
                # print(t_twt_id,"Already exists in list,ignoring...")
                continue
            if i<k_similar:
                sim_list[tr_twt_id]=get_cosine(t_twt_val["parsed_tweet"],tr_twt_val["parsed_tweet"])
                # sim_list[tr_twt_id]=get_jackard(t_twt_val["parsed_tweet"],tr_twt_val["parsed_tweet"])
                i=i+1
            else:
                new_sim_twt=get_cosine(t_twt_val["parsed_tweet"],tr_twt_val["parsed_tweet"])
                # new_sim_twt=get_jackard(t_twt_val["parsed_tweet"],tr_twt_val["parsed_tweet"])
                for sim_id,sim_val in sim_list.items():
                    if new_sim_twt > sim_val:
                        del sim_list[sim_id]
                        sim_list[tr_twt_id]=new_sim_twt
                        break
        k_sim_twts[t_twt_id]=sim_list
    return k_sim_twts

def sim_tweet_class_vote(train,test,sim_vals):
    print("Method: sim_tweet_class_vote(train,test,sim_vals)")
    for id,sim_dict in sim_vals.items():
        class_votes=[0] * n_classes
        for t_id in sim_dict.keys():
            for tr_cls in train[t_id]["classes"]:
                class_votes[tr_cls]=class_votes[tr_cls]+1
        test[id]["knn_votes"]=class_votes
    return class_votes

def find_word(tweet_text,word_list):
    tweet_text_blob=tb(tweet_text)
    word_count=0
    for term in word_list:
        if term in tweet_text_blob.words.lower():
            word_count=word_count+1
    return word_count

def contains_phone(text):
    phonePattern=re.compile(r'''
                # don't match beginning of string,number can start anywhere
    (\d{3})     # area code is 3 digits (e.g. '800')
    \D*         # optional separator is any number of non-digits
    (\d{3})     # trunk is 3 digits (e.g. '555')
    \D*         # optional separator
    (\d{4})     # rest of number is 4 digits (e.g. '1212')
    \D*         # optional separator
    (\d*)       # extension is optional and can be any number of digits
    $           # end of string
    ''',re.VERBOSE)
    # return len(phonePattern.findall(text))
    if len(phonePattern.findall(text)) > 0:
        return "phonenumber"
    else :
        return text

def contains(train,unique_words):
    print("Method: contains(train,unique_words)")
    units  =tb('litre liter kg kilogram gram packet kilometer meter pack sets ton meal equipment kit percentage')
    units  =units.words+units.words.pluralize()
    number =tb('lac lakh million thousand hundred')
    number =number.words+number.words.pluralize()
    ra     =tb('treat send sent sending supply offer distribute treat mobilize mobilized donate donated dispatch dispatched')
    ra     =ra.words+ra.words.pluralize()
    rr     =tb('need requirement require ranout shortage scarcity')
    rr     =rr.words+rr.words.pluralize()
    medical=tb('medicine hospital medical doctor injection syringe ambulance antibiotic')
    medical=medical.words+medical.words.pluralize()
    url    =tb('urlurl')
    phone  =tb('phonenumber')
    loc    =tb('at')

    feature_names = ['units','number','ra','rr','medical','loc','url','phone']
    feature_count_matrix=np.zeros((n_classes, (len(feature_names) + 1)))
    for id,vals in train.items():
        train[id]['units']  =find_word(vals["tweet_text"],units)
        train[id]['number'] =find_word(vals["tweet_text"],number)
        train[id]['ra']     =find_word(vals["tweet_text"],ra)
        train[id]['rr']     =find_word(vals["tweet_text"],rr)
        train[id]['medical']=find_word(vals["tweet_text"],medical)
        train[id]['loc']    =find_word(vals["tweet_text"],loc)
        train[id]['url']    =find_word(vals["tweet_text"],url)
        train[id]['phone']  =find_word(vals["tweet_text"],phone)
        train[id]['word']   =len(vals["parsed_tweet"].split())
        train[id]['char']   =len(vals["parsed_tweet"])-vals["parsed_tweet"].count(' ')
        train[id]['unique'] =unique_word_count_class(vals["parsed_tweet"],unique_words)
        train[id]['char_space']=len(vals["parsed_tweet"])
        for cls in train[id]['classes']:
            feature_count_matrix[cls][0] = feature_count_matrix[cls][0] + train[id]['units']
            feature_count_matrix[cls][1] = feature_count_matrix[cls][1] + train[id]['number']
            feature_count_matrix[cls][2] = feature_count_matrix[cls][2] + train[id]['ra']
            feature_count_matrix[cls][3] = feature_count_matrix[cls][3] + train[id]['rr']
            feature_count_matrix[cls][4] = feature_count_matrix[cls][4] + train[id]['medical']
            feature_count_matrix[cls][5] = feature_count_matrix[cls][5] + train[id]['loc']
            feature_count_matrix[cls][6] = feature_count_matrix[cls][6] + train[id]['url']
            feature_count_matrix[cls][7] = feature_count_matrix[cls][7] + train[id]['phone']
    np.set_printoptions(threshold=np.inf)
    print(feature_names)
    print(feature_count_matrix)

def unique_word_count_class(text,unique_words):
    cls_counts=[0] * n_classes
    for word in text.split():
        for cls in range(len(unique_words)):
            if word in unique_words[cls]:
                cls_counts[cls]=cls_counts[cls] + 1
    return cls_counts

def create_corpus(data,n_classes):
    print("Method: create_corpus(data,n_classes)")
    total_corpus=[]
    class_corpuses=dict((key,[]) for key in range(n_classes))
    for id,vals in data.items():
        total_corpus.append(vals["parsed_tweet"])
        class_corpuses[vals["classes"][0]].append(vals["parsed_tweet"])
    return total_corpus,class_corpuses

def most_freq_words(corpus,k_most_common):
    return FreqDist(corpus).most_common(k_most_common)

def tf(word,blob):
    '''computes "term frequency" which is the number of times a word appears in a document blob,
    normalized by dividing by the total number of words in blob.'''
    return blob.words.count(word) / len(blob.words)

def n_containing(word,bloblist):
    '''number of documents containing word'''
    return sum(1 for blob in bloblist if word in blob)

def idf(word,bloblist):
    '''computes "inverse document frequency" which measures how common a word is among all documents in bloblist. The more common a word is, the lower its idf'''
    return math.log(len(bloblist) / (1 + n_containing(word,bloblist)))

def tfidf(word,blob,bloblist):
    '''computes the TF-IDF score. It is simply the product of tf and idf'''
    return tf(word,blob) * idf(word,bloblist)

def get_cosine(tweet1,tweet2):
    '''calculates the cosine similarity between 2 tweets'''
    # print("Method: get_cosine(tweet1,tweet2)")
    from collections import Counter
    WORD=re.compile(r'\w+')
    vec1=Counter(WORD.findall(tweet1))
    vec2=Counter(WORD.findall(tweet2))

    intersection=set(vec1.keys()) & set(vec2.keys())
    numerator=sum([vec1[x] * vec2[x] for x in intersection])

    sum1=sum([vec1[x]**2 for x in vec1.keys()])
    sum2=sum([vec2[x]**2 for x in vec2.keys()])
    denominator=math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def unique_words_class(class_corpuses):
    ''' Finds unique words for each class'''
    print("Method: unique_words_class(class_corpuses)")
    bloblist=[]
    unique_words=defaultdict()
    for cls_id,text in class_corpuses.items():
        bloblist.append(tb(" ".join(text)))
    for i,blob in enumerate(bloblist):
        unique_words[i]=[]
        print("\nTop words in class {}".format(i))
        scores={word: tfidf(word,blob,bloblist) for word in blob.words}
        sorted_words=sorted(scores.items(),key=lambda x: x[1],reverse=True)
        for word,score in sorted_words[:k_unique_words]:
            print("{},TF-IDF: {}".format(word,round(score,5)))
            unique_words[i].append(word)
    return unique_words

def create_tf_idf(train,test,n_gram):
    '''Calculates tf-idf vectors for train and test'''
    print("Method: create_tf_idf(train,test)")
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer=TfidfVectorizer(strip_accents='unicode',decode_error='ignore',ngram_range=(1,n_gram))
    # tfidf_vectorizer=TfidfVectorizer(strip_accents='unicode',decode_error='ignore',ngram_range=(1,3))
    # tfidf_vectorizer=TfidfVectorizer(strip_accents='unicode',lowercase=True,max_features=10000,\
    # stop_words=stopwords.words('english'),decode_error='ignore',ngram_range=(1,3))
    train_tfidf_matrix=tfidf_vectorizer.fit_transform([vals["parsed_tweet"] for twt_id,vals in train.items()])
    test_tfidf_matrix =tfidf_vectorizer.transform([vals["parsed_tweet"] for twt_id,vals in test.items()])
    return train_tfidf_matrix,test_tfidf_matrix

# Supervised----------------------------------------------------------------------------------------
def supervised(train,test,train_tfidf_matrix,test_tfidf_matrix):
    print("Method: supervised(train,test,train_tfidf_matrix,test_tfidf_matrix)")
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.svm import LinearSVC
    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.multiclass import OneVsRestClassifier

    from sklearn.linear_model import SGDClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import PassiveAggressiveClassifier
    from sklearn.neighbors import KNeighborsClassifier

    result=OrderedDict()

    mlb=MultiLabelBinarizer()
    train_labels=[vals["classes"] for id,vals in train.items()]
    train_labels_bin=mlb.fit_transform(train_labels)
    test_labels=[vals["classes"] for id,vals in test.items()]

    print("\nAlgorithm: \t \t \t Adaboost")
    Adaboost =OneVsRestClassifier(AdaBoostClassifier(n_estimators=50)).fit(train_tfidf_matrix,train_labels_bin).predict(test_tfidf_matrix)
    accuracy_multi(test,test_labels,mlb.inverse_transform(Adaboost))
    result["Adaboost"]=sklearn_metrics(mlb.fit_transform(test_labels),Adaboost)

    print("\nAlgorithm: \t \t \t Decision_Tree")
    Decision_Tree =tree.DecisionTreeClassifier().fit(train_tfidf_matrix,train_labels_bin).predict(test_tfidf_matrix)
    accuracy_multi(test,test_labels,mlb.inverse_transform(Decision_Tree))
    result["Decision_Tree"]=sklearn_metrics(mlb.fit_transform(test_labels),Decision_Tree)

    print("\nAlgorithm: \t \t \t Naive_Bayes_Gaussian")
    Naive_Bayes_Gaussian =OneVsRestClassifier(GaussianNB()).fit(train_tfidf_matrix,train_labels_bin).predict(test_tfidf_matrix)
    accuracy_multi(test,test_labels,mlb.inverse_transform(Naive_Bayes_Gaussian))
    result["Naive_Bayes_Gaussian"]=sklearn_metrics(mlb.fit_transform(test_labels),Naive_Bayes_Gaussian)

    print("\nAlgorithm: \t \t \t SVM_Linear")
    SVM_Linear =OneVsRestClassifier(LinearSVC(random_state=0)).fit(train_tfidf_matrix,train_labels_bin).predict(test_tfidf_matrix)
    accuracy_multi(test,test_labels,mlb.inverse_transform(SVM_Linear))
    result["SVM_Linear"]=sklearn_metrics(mlb.fit_transform(test_labels),SVM_Linear)

    # print("\nAlgorithm: \t \t \t Adaboost_SVM")
    # Adaboost_SVM =OneVsRestClassifier(AdaBoostClassifier(LinearSVC(random_state=0),algorithm='SAMME',n_estimators=50)).fit(train_tfidf_matrix,train_labels_bin).predict(test_tfidf_matrix)
    # accuracy_multi(test,test_labels,mlb.inverse_transform(Adaboost_SVM))
    # result["Adaboost_SVM"]=sklearn_metrics(mlb.fit_transform(test_labels),Adaboost_SVM)

    print("\nAlgorithm: \t \t \t Random_Forest")
    Random_Forest =RandomForestClassifier().fit(train_tfidf_matrix,train_labels_bin).predict(test_tfidf_matrix)
    accuracy_multi(test,test_labels,mlb.inverse_transform(Random_Forest))
    result["Random_Forest"]=sklearn_metrics(mlb.fit_transform(test_labels),Random_Forest)

    print("\nAlgorithm: \t \t \t Gradient_Boosting")
    Gradient_Boosting=OneVsRestClassifier(GradientBoostingClassifier()).fit(train_tfidf_matrix,train_labels_bin).predict(test_tfidf_matrix)
    accuracy_multi(test,test_labels,mlb.inverse_transform(Gradient_Boosting))
    result["Gradient_Boosting"]=sklearn_metrics(mlb.fit_transform(test_labels),Gradient_Boosting)

    return result

# Accuracy------------------------------------------------------------------------------------------
def sklearn_metrics(actual,predicted):
    # print("Method: sklearn_metrics(actual,predicted)")
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import precision_recall_fscore_support

    results = OrderedDict()

    results["accuracy"] = accuracy_score(actual,predicted)
    results["precision_macro"] = precision_score(actual,predicted,average='macro')
    results["precision_micro"] = precision_score(actual,predicted,average='micro')
    results["recall_macro"] = recall_score(actual,predicted,average='macro')
    results["recall_micro"] = recall_score(actual,predicted,average='micro')
    results["f1_macro"] = f1_score(actual,predicted,average='macro')
    results["f1_micro"] = f1_score(actual,predicted,average='micro')
    results["Precision"] = precision_recall_fscore_support(actual,predicted)[0].tolist()
    results["Recall"] = precision_recall_fscore_support(actual,predicted)[1].tolist()
    results["F1"] = precision_recall_fscore_support(actual,predicted)[2].tolist()

    from termcolor import colored, cprint
    text = 'accuracy_score: '
    print(text,results["accuracy"])
    print("\t\t\t Macro,\t\t\t Micro")
    print("\t\t\t -----,\t\t\t -----")
    print("Precision:\t\t",results["precision_macro"],"\t",results["precision_micro"])
    print("Recall:\t\t\t",results["recall_macro"],"\t",results["recall_micro"])
    print("f1:\t\t\t",results["f1_macro"],"\t",results["f1_micro"])
    print(classification_report(y_true=actual,y_pred=predicted,target_names=class_names,digits=4))
    print("\n")
    return results

def accuracy_multi(all,actual,predicted,multi=True):
    '''Calculates (Macro,Micro) precision,recall'''
    # print("Method: accuracy_multi(all,actual,predicted,multi=True)")
    if len(actual) != len(predicted):
        print("** length does not match: ",len(actual),len(predicted))
    class_count=[0] * n_classes
    for i in range(len(actual)):
        if multi:
            for pred_label in predicted[i]:
                if pred_label in actual[i]:
                    class_count[pred_label]=class_count[pred_label]+1
        else:
            if actual[i] == predicted[i]:
                class_count[predicted[i]]=class_count[predicted[i]]+1
    print("Predicted counts per class:\t",class_count)

def split_data(lab_tweets,test_size):
    ''' splits the data based on test_size'''
    print("Method: split_data(lab_tweets,test_size)")
    from sklearn.model_selection import train_test_split
    all_list=list(lab_tweets.keys())
    train_split,test_split=train_test_split(all_list,test_size=test_size)
    train=OrderedDict()
    test=OrderedDict()
    for id in train_split:
        train[id]=lab_tweets[id]
    for id in test_split:
        test[id]=lab_tweets[id]
    return train,test

def add_features_matrix(train,train_matrix,manual=False,length=True):
    print("Method: add_features_matrix(train,train_matrix,lengths=False)")

    if manual:
        print("\n Manual features...\n")

        loc = np.matrix([[val["loc"] / val["word"]] for id,val in train.items()])
        new = np.concatenate((train_matrix,loc), axis=1)

        medical = np.matrix([[val["medical"] / val["word"]] for id,val in train.items()])
        new = np.concatenate((new,medical), axis=1)

        number = np.matrix([[val["number"] / val["word"]] for id,val in train.items()])
        new = np.concatenate((new,number), axis=1)

        ra = np.matrix([[val["ra"] / val["word"]] for id,val in train.items()])
        new = np.concatenate((new,ra), axis=1)

        rr = np.matrix([[val["rr"] / val["word"]] for id,val in train.items()])
        new = np.concatenate((new,rr), axis=1)

        units = np.matrix([[val["units"] / val["word"]] for id,val in train.items()])
        new = np.concatenate((new,units), axis=1)

    else :
        print("\n Atomatic features...\n")

        retweet_count_max = max([val["retweet_count"] for id,val in train.items()])
        retweet_count = np.matrix([[val["retweet_count"] / retweet_count_max] for id,val in train.items()])
        new = np.concatenate((train_matrix,retweet_count), axis=1)

        url = np.matrix([[val["url"] / val["word"]] for id,val in train.items()])
        new = np.concatenate((new,url), axis=1)

        phone = np.matrix([[val["phone"] / val["word"]] for id,val in train.items()])
        new = np.concatenate((new,phone), axis=1)

        for i in range(n_classes):
            unique = np.matrix([val["unique"][i] / k_unique_words for id,val in train.items()])
            new = np.concatenate((new,unique.T), axis=1)

        for i in range(n_classes):
            knn_votes = np.matrix([val["knn_votes"][i] / k_unique_words for id,val in train.items()])
            new = np.concatenate((new,knn_votes.T), axis=1)

        if length:
            print("\n Length related features...\n")

            char_max = max([val["char"] for id,val in train.items()])
            char = np.matrix([[(val["char"] / char_max)] for id,val in train.items()])
            new = np.concatenate((train_matrix,char), axis=1)

            char_space_max = max([val["char_space"] for id,val in train.items()])
            char_space = np.matrix([[(val["char_space"] / char_space_max)] for id,val in train.items()])
            new = np.concatenate((new,char_space), axis=1)

            word_max = max([val["word"] for id,val in train.items()])
            word = np.matrix([[val["word"] / word_max] for id,val in train.items()])
            new = np.concatenate((new,word), axis=1)

    return new

from plotly.graph_objs import *
import plotly.plotly as py
def plot(results, iter, username, key):
    print("Method: plot(results, iter)")

    py.sign_in(username, key)

    metrics = ['Precision', 'Recall', 'F1']
    for index,metric in enumerate(metrics):
        class_values = []
        for algo,values in results.items():
            class_values.append({
              "x": ["1","2","3","4","5","6","7",],
              "y": values[metric],
              "name": algo,
              "type": "bar",
              "xaxis": "x",
              "yaxis": "y",
            })
        data = Data(class_values)
        
        layout = {
            "autosize": True, 
            "barmode": "group",
            "dragmode": "zoom",
            "hovermode": "x",
            "legend": {"orientation": "h"},
            "margin": {
                "t": 40,
                "b": 110
            },
            "showlegend": True,
            "title": metric,
            "xaxis": {
                "anchor": "y",
                "autorange": True,
                "domain": [-1, 1],
                "dtick": 1, 
                "exponentformat": "none", 
                "fixedrange": False,
                "nticks": 1,
                "range": [-0.5, 6.5],
                "showgrid": False,
                "showline": False,
                "showticklabels": True,
                "ticks": "",
                "title": "<b>Classes</b>",
                "type": "category",
                "zeroline": False
            },
            "yaxis": {
                "anchor": "x",
                "autorange": True, 
                "dtick": 0.05, 
                "range": [0, 1],
                "showgrid": False,
                "tick0": 0, 
                "tickangle": "auto", 
                "tickmode": "linear", 
                "tickprefix": "", 
                "ticks": "",
                "title": "<b>Values</b>",
                "type": "linear",
                "zeroline": False
          },
        }
        fig = Figure(data=data, layout=layout)        
        filename=metric+'_'+iter
        print(filename)
        try:
            py.image.save_as(fig, filename=filename+'.png')
            plot_url = py.plot(fig,filename= metric)
        except Exception as e:
            print("Could not plot graph. Failure reason: ",e)

def features(train,test):
    sim_vals_train=k_similar_tweets(train,train,k_similar)
    sim_vals_test=k_similar_tweets(train,test,k_similar)
    sim_tweet_class_vote(train,train,sim_vals_train)
    sim_tweet_class_vote(train,test,sim_vals_test)

    total_corpus,class_corpuses=create_corpus(train,n_classes)
    unique_words=unique_words_class(class_corpuses) # TODO: unique word list can be increased by iteration on test data as followed in "SMERP paper 1"

    contains(train,unique_words)
    contains(test,unique_words)

def write_file(data,file_name):
    with open(file_name,"w", encoding="utf-8") as out_file:
        out_file.write(str(data))
    out_file.close()

def parse_tweets(train):
    print("Method: parse_tweets(train)")
    for id,val in train.items():
        val['parsed_tweet'] = parse_tweet(val['tweet_text'])
    return train

# Main----------------------------------------------------------------------------------------------
def main():
    print("Method: main()")
    algo_list=["Adaboost","Decision_Tree","Gradient_Boosting","Naive_Bayes_Gaussian","Random_Forest","SVM_Linear"]

    test_size=0.3 # portion of the data to be used in test
    username = '' # plotly username
    key      = '' # plotly key
    
    lab_tweets=read_json('labelled_tweets')
    lab_tweets= parse_tweets(lab_tweets)
    train,test=split_data(lab_tweets,test_size)
    print("train size:",len(train))
    print("test size:",len(test))
    
    train_tfidf_matrix_1,test_tfidf_matrix_1=create_tf_idf(train,test,1)
    train_tfidf_matrix_1 = train_tfidf_matrix_1.todense()
    test_tfidf_matrix_1 = test_tfidf_matrix_1.todense()
    train_tfidf_matrix_2,test_tfidf_matrix_2=create_tf_idf(train,test,2)
    train_tfidf_matrix_2 = train_tfidf_matrix_2.todense()
    test_tfidf_matrix_2 = test_tfidf_matrix_2.todense()

    ## 1. tf-idf
    fs1_unigram = supervised(train,test,train_tfidf_matrix_1,test_tfidf_matrix_1)
    save_json(fs1_unigram,"fs1_unigram")
    plot(fs1_unigram, "fs1_unigram" ,username, key)

    ## 2. tf-idf + bigrams
    fs2_bigrams = supervised(train,test,train_tfidf_matrix_2,test_tfidf_matrix_2)
    save_json(fs2_bigrams,"fs2_bigrams")
    plot(fs2_bigrams, "fs2_bigrams" ,username, key)

    ## features
    features(train,test)

    ## 3. unigrams + features
    train_tf_idf1_manual=add_features_matrix(train,train_tfidf_matrix_1,manual=True)
    test_tf_idf1_manual=add_features_matrix(test,test_tfidf_matrix_1,manual=True)
    fs3_manual_1 =  supervised(train,test,train_tf_idf1_manual,test_tf_idf1_manual)
    save_json(fs3_manual_1,"fs3_manual_1")
    plot(fs3_manual_1, "fs3_manual_1" ,username, key)

    ## 4. bigrams + manual
    train_tf_idf2_manual=add_features_matrix(train,train_tfidf_matrix_2,manual=True)
    test_tf_idf2_manual=add_features_matrix(test,test_tfidf_matrix_2,manual=True)
    fs4_manual_2 =  supervised(train,test,train_tf_idf2_manual,test_tf_idf2_manual)
    save_json(fs4_manual_2,"fs4_manual_2")
    plot(fs4_manual_2, "fs4_manual_2" ,username, key)

    ## 4. unigrams + auto
    train_tf_idf1_auto=add_features_matrix(train,train_tfidf_matrix_1)
    test_tf_idf1_auto=add_features_matrix(test,test_tfidf_matrix_1)
    fs5_auto_1 =  supervised(train,test,train_tf_idf1_auto,test_tf_idf1_auto)
    save_json(fs5_auto_1,"fs5_auto_1")
    plot(fs5_auto_1, "fs5_auto_1" ,username, key)

    ## 4. bigrams + auto
    train_tf_idf2_auto=add_features_matrix(train,train_tfidf_matrix_2)
    test_tf_idf2_auto=add_features_matrix(test,test_tfidf_matrix_2)
    fs6_auto_2 =  supervised(train,test,train_tf_idf2_auto,test_tf_idf2_auto)
    save_json(fs6_auto_2,"fs6_auto_2")
    plot(fs6_auto_2, "fs6_auto_2" ,username, key)

    ## 4. unigrams + both
    train_tf_idf1_both=add_features_matrix(train,train_tf_idf1_auto,manual=True)
    test_tf_idf1_both=add_features_matrix(test,test_tf_idf1_auto,manual=True)
    fs7_both_1 =  supervised(train,test,train_tf_idf1_both,test_tf_idf1_both)
    save_json(fs7_both_1,"fs7_both_1")
    plot(fs7_both_1, "fs7_both_1" ,username, key)
    
    ## 4. bigrams + both
    train_tf_idf2_both=add_features_matrix(train,train_tf_idf2_auto,manual=True)
    test_tf_idf2_both=add_features_matrix(test,test_tf_idf2_auto,manual=True)
    fs8_both_2 =  supervised(train,test,train_tf_idf2_both,test_tf_idf2_both)
    save_json(fs8_both_2,"fs8_both_2")
    plot(fs8_both_2, "fs8_both_2" ,username, key)

    ## save the input set for verification
    save_json(train,"train_mod")
    save_json(test,"test_mod")

if __name__ == "__main__": main()
