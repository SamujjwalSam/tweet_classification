# coding=utf-8
# !/usr/bin/python
'''
INFO:
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
    # print(tweet)
    stop=stopwords.words('english') + list(string.punctuation) + ['rt','via','& amp']
    tweet=re.sub(r"http\S+","urlurl",tweet) # replaces hyperlink with urlurl
    terms=preprocess(tweet,True)
    for term_pos in range(len(terms)):
        terms[term_pos]=terms[term_pos].replace("@","")
        terms[term_pos]=terms[term_pos].replace("#","")
        terms[term_pos]=get_acronyms(terms[term_pos])
        terms[term_pos]=contains_phone(terms[term_pos])
        #TODO: pre-process the acronym
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
    return blob.words.count(word) / len(blob.words)

def n_containing(word,bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word,bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word,bloblist)))

def tfidf(word,blob,bloblist):
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

def create_class_cluster_matrix(train,n_classes,n_clusters):
    print("Method: create_class_cluster_matrix(train,n_clusters)")
    matrix=[[0 for x in range(n_clusters)] for y in range(7)]
    for cls_no in range(0,n_classes):
        for clst_no in range(0,n_clusters):
            matrix [cls_no][clst_no]=[tweet_obj.cluster_no for tweet_obj in train if tweet_obj.class_no == cls_no].count(clst_no) / \
            len([tweet_obj.id for tweet_obj in train if tweet_obj.cluster_no == clst_no])
            print("cls_no= ",cls_no," clst_no= ",clst_no," match: ",[tweet_obj.cluster_no for tweet_obj in train if tweet_obj.class_no == cls_no].count(clst_no)," cluster_count: ",len([tweet_obj.id for tweet_obj in train if tweet_obj.cluster_no == clst_no])," Weight: ",[tweet_obj.cluster_no for tweet_obj in train if tweet_obj.class_no == cls_no].count(clst_no) / len([tweet_obj.id for tweet_obj in train if tweet_obj.cluster_no == clst_no]))
    return matrix

# Unsupervised--------------------------------------------------------------------------------------
def kmeans_fit(W,n_clusters):
    print("Method: kmeans_fit(W)")
    kmeans=cluster.KMeans(n_clusters=n_clusters)
    kmeans.fit(W)
    kmeans_pred_labels=kmeans.predict(W)
    for clus_no in range (1,n_clusters):
        print("K-Means Cluster points %d: "%clus_no)
        print(len(np.where(kmeans_pred_labels == clus_no)[0]))
        print(np.where(kmeans_pred_labels == clus_no)[0])
    return kmeans

def kmeans_predict(k,centroids_t):
    print("Method: kmeans_predict(k,centroids_t)")
    indices=[]
    k_smallest=[]
    for tweet in centroids_t:
        k_smallest=heapq.nsmallest(k,tweet)
        k_smallest_indices=[]
        for k_small in k_smallest:
            k_smallest_indices.append(np.where(tweet == k_small))
        indices.append(k_smallest_indices)
    return indices

def spectral(W,n_clusters):
    print("Method: spectral(W)")
    global id_cluster
    global id_tweets_sliced
    global labeled_id_tweets
    spectral_clusters=cluster.spectral_clustering(W,n_clusters=n_clusters,eigen_solver='arpack')
    for clus_no in range (0,n_clusters):
        print("Spectral Cluster points %d: "%clus_no)
        print(len(np.where(spectral_clusters == clus_no)[0]))
        print(np.where(spectral_clusters == clus_no)[0])
    return spectral_clusters

def lda(W,n_clusters):
    print("Method: lda(W,n_clusters)")
    from gensim import corpora,models

def lsh(W,n_clusters):
    print("Method: lsh(W,n_clusters)")
    from sklearn.neighbors import LSHForest
    LSHForest(random_state=42).fit(W)

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

    # print("\nAlgorithm: \t \t \t PassiveAggressiveClassifier")
    # PassiveAggressiveClassifier =OneVsRestClassifier(PassiveAggressiveClassifier()).fit(train_tfidf_matrix,train_labels_bin).predict(test_tfidf_matrix)
    # accuracy_multi(test,test_labels,mlb.inverse_transform(PassiveAggressiveClassifier))
    # result["PassiveAggressiveClassifier"]=sklearn_metrics(mlb.fit_transform(test_labels),PassiveAggressiveClassifier)

    # print("\nAlgorithm: \t \t \t SGDClassifier")
    # SGDClassifier =OneVsRestClassifier(SGDClassifier()).fit(train_tfidf_matrix,train_labels_bin).predict(test_tfidf_matrix)
    # accuracy_multi(test,test_labels,mlb.inverse_transform(SGDClassifier))
    # result["SGDClassifier"]=sklearn_metrics(mlb.fit_transform(test_labels),SGDClassifier)

    # print("\nAlgorithm: \t \t \t Perceptron")
    # Perceptron =OneVsRestClassifier(Perceptron()).fit(train_tfidf_matrix,train_labels_bin).predict(test_tfidf_matrix)
    # accuracy_multi(test,test_labels,mlb.inverse_transform(Perceptron))
    # result["Perceptron"]=sklearn_metrics(mlb.fit_transform(test_labels),Perceptron)

    # print("\nAlgorithm: \t \t \t KNeighborsClassifier")
    # KNeighborsClassifier =OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10)).fit(train_tfidf_matrix,train_labels_bin).predict(test_tfidf_matrix)
    # accuracy_multi(test,test_labels,mlb.inverse_transform(KNeighborsClassifier))
    # result["KNeighborsClassifier"]=sklearn_metrics(mlb.fit_transform(test_labels),KNeighborsClassifier)

    # print("\nAlgorithm: \t \t \t Naive_Bayes_Multinomial")
    # naive_bayes_m =OneVsRestClassifier(MultinomialNB()).fit(train_tfidf_matrix,train_labels_bin).predict(test_tfidf_matrix)
    # accuracy_multi(test,test_labels,mlb.inverse_transform(naive_bayes_m))
    # result["Naive_Bayes_Multinomial"]=sklearn_metrics(mlb.fit_transform(test_labels),naive_bayes_m)

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

    # print("\nAlgorithm: \t \t \t Random_Forest")
    # Random_Forest =RandomForestClassifier().fit(train_tfidf_matrix,train_labels_bin).predict(test_tfidf_matrix)
    # accuracy_multi(test,test_labels,mlb.inverse_transform(Random_Forest))
    # result["Random_Forest"]=sklearn_metrics(mlb.fit_transform(test_labels),Random_Forest)

    # print("\nAlgorithm: \t \t \t Gradient_Boosting")
    # Gradient_Boosting=OneVsRestClassifier(GradientBoostingClassifier()).fit(train_tfidf_matrix,train_labels_bin).predict(test_tfidf_matrix)
    # accuracy_multi(test,test_labels,mlb.inverse_transform(Gradient_Boosting))
    # result["Gradient_Boosting"]=sklearn_metrics(mlb.fit_transform(test_labels),Gradient_Boosting)

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
    results["precision"] = precision_recall_fscore_support(actual,predicted)[0].tolist()
    results["recall"] = precision_recall_fscore_support(actual,predicted)[1].tolist()
    results["f1"] = precision_recall_fscore_support(actual,predicted)[2].tolist()

    from termcolor import colored, cprint
    # text = colored('accuracy_score: ', 'green', attrs=['blink'])
    text = 'accuracy_score: '
    print(text,'\x1b[1;31m',results["accuracy"],'\x1b[0m')
    print("\t\t\t Macro,\t\t\t Micro")
    print("\t\t\t -----,\t\t\t -----")
    print("precision:\t\t",results["precision_macro"],"\t",results["precision_micro"])
    print("recall:\t\t\t",results["recall_macro"],"\t",results["recall_micro"])
    print("f1:\t\t\t",results["f1_macro"],"\t",results["f1_micro"])
    # print("precision: ",results["precision"])
    # print("recall: ",results["recall"])
    # print("f1: ",results["f1"])
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

def feature_selection(train,test,train_matrix,test_matrix):
    print("Method: feature_selection(train,test,train_matrix,test_matrix)")
    # from sklearn.preprocessing import MultiLabelBinarizer
    # mlb=MultiLabelBinarizer()
    train_labels=[vals["classes"][0] for id,vals in train.items()]
    # train_labels_bin=mlb.fit_transform(train_labels)
    from sklearn.feature_selection import SelectPercentile
    sel = SelectPercentile(percentile=20)
    train_matrix_vt=sel.fit_transform(train_matrix,train_labels)
    test_matrix_vt=sel.transform(test_matrix)
    print(train_matrix_vt.shape)
    print(test_matrix_vt.shape)

def use_pca(train,test,train_matrix,test_matrix,pca_increment=200,add_features=True):
    '''Computer PCA on tf-idf matrices'''
    print("Method: use_pca(train,test,train_matrix,test_matrix,pca_increment=200,add_features=True)")
    from sklearn.decomposition import PCA
    pca_result = OrderedDict() # to store the results
    run_n=int(train_matrix.shape[1]/pca_increment)
    for i in range(run_n):
        if i == (run_n-1):
            print("Feature_count (all): ",train_matrix.shape[1])
            pca_result[train_matrix.shape[1]]=supervised(train,test,train_matrix,test_matrix)
            break
        n_features=(i+1) * pca_increment
        print("Feature_count: ", n_features)
        pca = PCA(n_components=n_features)
        train_matrix_pca=pca.fit_transform(train_matrix)
        test_matrix_pca=pca.transform(test_matrix)
        if add_features: # add extra features to the matrix
            train_features=add_features_matrix(train,train_matrix_pca)
            test_features=add_features_matrix(test,test_matrix_pca)
            pca_result[n_features]=supervised(train,test,train_features,test_features)
        else: # run algos only on tf-idf
            pca_result[n_features]=supervised(train,test,train_matrix_pca,test_matrix_pca)
    return pca_result

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
def plot(results, iter):
    print("Method: plot(results, iter)")

    py.sign_in('samujjwal86', 'U3gIQsZHKYNN5q3fqKF0')

    metrics = ['precision', 'recall', 'f1']
    for index,metric in enumerate(metrics):
        class_values = []
        for algo,values in results.items():
            class_values.append({
              "x": class_names,
              "y": values[metric],
              "name": algo,
              "type": "bar",
              "xaxis": "x",
              "yaxis": "y",
            })
        data = Data(class_values)
        layout = {
          "barmode": "group",
          "dragmode": "zoom",
          "hovermode": "closest",
          "margin": {
            "t": 40,
            "b": 110
          },
          "showlegend": True,
          "title": metric+'_'+iter,
          "xaxis": {
            "anchor": "y",
            "autorange": False,
            "domain": [-1, 1],
            "fixedrange": False,
            "nticks": 1,
            "range": [-1, 8],
            "showgrid": False,
            "showline": False,
            "showticklabels": False,
            "ticks": "",
            "title": "<b>Classes</b>",
            "type": "-",
            "zeroline": False
          },
          "yaxis": {
            "anchor": "x",
            "autorange": False,
            "range": [0, 1],
            "showgrid": False,
            "ticks": "",
            "title": "<b>Values</b>",
            "type": "linear",
            "zeroline": False
          },
        }
        fig = Figure(data=data, layout=layout)
        ## plot_url = py.plot(fig)
        filename=metric+'_'+iter+'.png'
        print(filename)
        py.image.save_as(fig, filename=filename)

def tables():
    print("Method: tables(results_all)")
    import plotly.plotly as py
    from plotly.tools import FigureFactory as ff
    py.sign_in('samujjwal86', 'U3gIQsZHKYNN5q3fqKF0')

    metric_matrix = [["Precision_1","Naive_Bayes_Gaussian","SVM_Linear","Decision_Tree","Adaboost"],
                   ['tf-idf', 0.68862275, 0.82978723, 0.61797753, 0.67469879],
                   ['tf-idf + features', 0.68862275, 0.83098592, 0.71830986, 0.74011299],
                   ['tf-idf + features + lengths', 0.68862275, 0.82394366, 0.71612903, 0.74011299]]
    table = ff.create_table(metric_matrix, index=True)
    # plot_url = py.plot(table, filename='Precision_1')
    py.image.save_as(table, filename='Precision_1.png')

    metric_matrix = [["Precision_2","Naive_Bayes_Gaussian","SVM_Linear","Decision_Tree","Adaboost"],
                   ['tf-idf', 0.7291666666666666, 0.8857142857142857, 0.8, 0.7816091954022989],
                   ['tf-idf + features', 0.7291666666666666, 0.88, 0.7209302325581395, 0.7654320987654321],
                   ['tf-idf + features + lengths', 0.7291666666666666, 0.8783783783783784, 0.7530864197530864, 0.7654320987654321]]
    table = ff.create_table(metric_matrix, index=True)
    # plot_url = py.plot(table, filename='Precision_2')
    py.image.save_as(table, filename='Precision_2.png')

    metric_matrix = [["Precision_3","Naive_Bayes_Gaussian","SVM_Linear","Decision_Tree","Adaboost"],
                   ['tf-idf', 0.6753246753246753, 0.8314606741573034, 0.7472527472527473, 0.7238095238095238],
                   ['tf-idf + features', 0.6753246753246753, 0.8085106382978723, 0.660377358490566, 0.719626168224299],
                   ['tf-idf + features + lengths', 0.6753246753246753, 0.7572815533980582, 0.673469387755102, 0.7181818181818181]]
    table = ff.create_table(metric_matrix, index=True)
    # plot_url = py.plot(table, filename='Precision_3')
    py.image.save_as(table, filename='Precision_3.png')

    metric_matrix = [["Precision_4","Naive_Bayes_Gaussian","SVM_Linear","Decision_Tree","Adaboost"],
                   ['tf-idf', 0.5, 0.75, 0.48, 0.55],
                   ['tf-idf + features', 0.5, 0.6, 0.32, 0.35294117647058826],
                   ['tf-idf + features + lengths', 0.5, 0.6, 0.37037037037037035, 0.35294117647058826]]
    table = ff.create_table(metric_matrix, index=True)
    # plot_url = py.plot(table, filename='Precision_4')
    py.image.save_as(table, filename='Precision_4.png')

    metric_matrix = [["Precision_5","Naive_Bayes_Gaussian","SVM_Linear","Decision_Tree","Adaboost"],
                   ['tf-idf', 0.5666666666666667, 0.95, 0.32608695652173914, 0.45454545454545453],
                   ['tf-idf + features', 0.5666666666666667, 0.8695652173913043, 0.4186046511627907, 0.4358974358974359],
                   ['tf-idf + features + lengths', 0.5666666666666667, 0.8333333333333334, 0.44, 0.40384615384615385]]
    table = ff.create_table(metric_matrix, index=True)
    # plot_url = py.plot(table, filename='Precision_5')
    py.image.save_as(table, filename='Precision_5.png')

    metric_matrix = [["Precision_6","Naive_Bayes_Gaussian","SVM_Linear","Decision_Tree","Adaboost"],
                   ['tf-idf', 0.5730337078651685, 0.6103896103896104, 0.4144144144144144, 0.5353535353535354],
                   ['tf-idf + features', 0.5730337078651685, 0.5833333333333334, 0.45544554455445546, 0.5463917525773195],
                   ['tf-idf + features + lengths', 0.5730337078651685, 0.5764705882352941, 0.4224137931034483, 0.4789915966386555]]
    table = ff.create_table(metric_matrix, index=True)
    # plot_url = py.plot(table, filename='Precision_6')
    py.image.save_as(table, filename='Precision_6.png')

    metric_matrix = [["Precision_7","Naive_Bayes_Gaussian","SVM_Linear","Decision_Tree","Adaboost"],
                   ['tf-idf', 0.9387755102040817, 0.9818181818181818, 0.5888888888888889, 0.8620689655172413],
                   ['tf-idf + features', 0.9387755102040817, 0.9, 0.8208955223880597, 0.8142857142857143],
                   ['tf-idf + features + lengths', 0.9387755102040817, 0.9152542372881356, 0.8181818181818182, 0.8461538461538461]]
    table = ff.create_table(metric_matrix, index=True)
    # plot_url = py.plot(table, filename='Precision_7')
    py.image.save_as(table, filename='Precision_7.png')

    # metrics = ['Precision', 'Recall', 'F1']

    # for cls in range(n_classes):
        # for i,metric in enumerate(metrics):
            # m_vals = [[0 for x in range(n_clusters)] for y in range(7)]
            # for fset,values in results_all.items():
                # for algo,vals in values.items():
                    # algos = values.keys()
                        # m_vals[fset].append(vals["precision_recall_fscore_support"][cls])
                # metric_matrix = [[metric+algos],
                               # [fset, m_vals],]

        # table = ff.create_table(metric_matrix, index=True)
        #### plot_url = py.plot(table, filename='simple_table.png')
        # py.image.save_as(table, filename=str(cls)+metric+'_table.png')

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

    # pca_increment=200
    # test_size =0.3
    # lab_tweets=read_json('labelled_tweets')
    # train,test=split_data(lab_tweets,test_size)
    # print("train size:",len(train))
    # print("test size:",len(test))

    train=read_json('labelled_tweets_train')
    test=read_json('labelled_tweets_test')

    train = parse_tweets(train)
    test = parse_tweets(test)
    
    # train count: [401, 210, 231, 75, 135, 252, 178]

    train_tfidf_matrix_1,test_tfidf_matrix_1=create_tf_idf(train,test,1)
    train_tfidf_matrix_1 = train_tfidf_matrix_1.todense()
    test_tfidf_matrix_1 = test_tfidf_matrix_1.todense()
    train_tfidf_matrix_2,test_tfidf_matrix_2=create_tf_idf(train,test,2)
    train_tfidf_matrix_2 = train_tfidf_matrix_2.todense()
    test_tfidf_matrix_2 = test_tfidf_matrix_2.todense()

    ## feature selection and PCA
    ## feature_selection(train,test,train_tfidf_matrix,test_tfidf_matrix)

    ## results=use_pca(train,test,train_tfidf_matrix,test_tfidf_matrix,pca_increment, False)
    ## print(results)
    ## features_result=use_pca(train,test,train_features,test_features,pca_increment)
    ## print(features_result)

    ## 1. tf-idf
    unigram = supervised(train,test,train_tfidf_matrix_1,test_tfidf_matrix_1)
    save_json(unigram,"fs1_unigram")

    ## 2. tf-idf + bigrams
    bigrams = supervised(train,test,train_tfidf_matrix_2,test_tfidf_matrix_2)
    save_json(bigrams,"fs2_bigrams")

    ## features
    features(train,test)

    ## 3. unigrams + features
    # train_tf_idf1_manual=add_features_matrix(train,train_tfidf_matrix_1,manual=True)
    # test_tf_idf1_manual=add_features_matrix(test,test_tfidf_matrix_1,manual=True)
    # manual_1 =  supervised(train,test,train_tf_idf1_manual,test_tf_idf1_manual)
    # save_json(manual_1,"fs3_manual_1")

    ## 4. bigrams + manual
    train_tf_idf2_manual=add_features_matrix(train,train_tfidf_matrix_2,manual=True)
    test_tf_idf2_manual=add_features_matrix(test,test_tfidf_matrix_2,manual=True)
    manual_2 =  supervised(train,test,train_tf_idf2_manual,test_tf_idf2_manual)
    save_json(manual_2,"fs4_manual_2")

    ## 4. unigrams + auto
    train_tf_idf1_auto=add_features_matrix(train,train_tfidf_matrix_1)
    test_tf_idf1_auto=add_features_matrix(test,test_tfidf_matrix_1)
    auto_1 =  supervised(train,test,train_tf_idf1_auto,test_tf_idf1_auto)
    save_json(auto_1,"fs5_auto_1")

    ## 4. bigrams + auto
    train_tf_idf2_auto=add_features_matrix(train,train_tfidf_matrix_2)
    test_tf_idf2_auto=add_features_matrix(test,test_tfidf_matrix_2)
    auto_2 =  supervised(train,test,train_tf_idf2_auto,test_tf_idf2_auto)
    save_json(auto_2,"fs6_auto_2")

    ## 4. unigrams + both
    train_tf_idf1_both=add_features_matrix(train,train_tf_idf1_auto,manual=True)
    test_tf_idf1_both=add_features_matrix(test,test_tf_idf1_auto,manual=True)
    both_1 =  supervised(train,test,train_tf_idf1_both,test_tf_idf1_both)
    save_json(both_1,"fs7_both_1")
    
    ## 4. bigrams + both
    train_tf_idf2_both=add_features_matrix(train,train_tf_idf2_auto,manual=True)
    test_tf_idf2_both=add_features_matrix(test,test_tf_idf2_auto,manual=True)
    both_2 =  supervised(train,test,train_tf_idf2_both,test_tf_idf2_both)
    save_json(both_2,"fs8_both_2")

    ## 5. tf-idf + unigrams + features - lengths
    train_tf_idf1_features_lengths=add_features_matrix(train,train_tf_idf1_manual,manual=True,length=False)
    test_tf_idf1_features_lengths=add_features_matrix(test,test_tf_idf1_manual,manual=True,length=False)
    fs9_both_1_length = supervised(train,test,train_tf_idf1_features_lengths,test_tf_idf1_features_lengths)
    save_json(fs9_both_1_length,"fs9_both_1_length")
    
    ## 6. tf-idf + bigrams + features - lengths
    train_tf_idf2_features_lengths=add_features_matrix(train,train_tf_idf2_manual,manual=True,length=False)
    test_tf_idf2_features_lengths=add_features_matrix(test,test_tf_idf2_manual,manual=True,length=False)
    fs10_both_2_length = supervised(train,test,train_tf_idf2_features_lengths,test_tf_idf2_features_lengths)
    save_json(fs10_both_2_length,"fs10_both_2_length")

    ## save the input set for verification
    save_json(train,"train_mod")
    save_json(test,"test_mod")

if __name__ == "__main__": main()