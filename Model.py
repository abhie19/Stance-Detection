# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 18:42:32 2016

@author: alienware
"""
########################3reading data [RUN THIS]############################3
import sklearn
import nltk
import numpy as np
import pandas as pd
import os
from scipy.sparse import hstack
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer

train = pd.read_excel(os.getcwd()+'\\train.xlsx')
test = pd.read_excel(os.getcwd()+'\\test.xlsx')
########################3reading data############################3

## NER and POS maker############################################################################

def pos_tag(series):
    import nltk
    def rem_mentions_hasht(tweet):
        words = tweet.split()
        relevant_tokens = [w for w in words if '@' not in w and '#' not in w]
        return( " ".join(relevant_tokens))
    
    series = series.apply(lambda tweet: rem_mentions_hasht(tweet))

    from nltk.tag.stanford import StanfordPOSTagger
    import os
    java_path = "C:/Program Files/Java/jre1.8.0_111/bin/java.exe"
    os.environ['JAVAHOME'] = java_path
    
    english_postagger = StanfordPOSTagger(os.getcwd()+'\\stanford-postagger-full-2014-08-27\\models\\english-bidirectional-distsim.tagger'
    , os.getcwd()+'\\stanford-postagger-full-2014-08-27\\stanford-postagger.jar')
    
    return series.apply(lambda a: english_postagger.tag(nltk.word_tokenize(a)))


def ner_tag(series):
    import nltk
    def rem_mentions_hasht(tweet):
        words = tweet.split()
        relevant_tokens = [w for w in words if '@' not in w and '#' not in w]
        return( " ".join(relevant_tokens))
    def ner_tagging(sent):
        try:
            return english_nertagger.tag(nltk.word_tokenize(sent))
        except:
            return []                                                               
    
    series = series.apply(lambda tweet: rem_mentions_hasht(tweet))

    from nltk.tag.stanford import StanfordNERTagger
    import os
    java_path = "C:/Program Files/Java/jre1.8.0_111/bin/java.exe"
    os.environ['JAVAHOME'] = java_path
    
    english_nertagger = StanfordNERTagger(os.getcwd()+'/stanford-ner-2014-08-27/classifiers/english.all.3class.distsim.crf.ser.gz', os.getcwd()+'/stanford-ner-2014-08-27/stanford-ner.jar')
    
    
    return series.apply(lambda a: ner_tagging(a))

train_stfrd_NERtagged = ner_tag(train.Tweet)
print("done train NER")
test_stfrd_NERtagged = ner_tag(test.Tweet)
print("done test NER")
test_stfrd_POStagged = pos_tag(test.Tweet)
print("done test POS")
train_stfrd_POStagged = pos_tag(train.Tweet)
print("done test POS")

from helper_funcs import *
 


import pickle
pickle.dump(train_stfrd_POStagged,open("train_stfrd_POStagged.p","wb"))
pickle.dump(test_stfrd_POStagged,open("test_stfrd_POStagged.p","wb"))
pickle.dump(train_stfrd_NERtagged,open("train_stfrd_NERtagged.p","wb"))
pickle.dump(test_stfrd_NERtagged,open("test_stfrd_NERtagged.p","wb"))

pickle.dump(create_arguing_features(train.Tweet), open("train_arg_lex.p","wb"))
pickle.dump(create_arguing_features(test.Tweet), open("test_arg_lex.p","wb"))

pickle.dump(mpqa_features(train_stfrd_POStagged), open("train_mpqa.p","wb"))
pickle.dump(mpqa_features(test_stfrd_POStagged), open("test_mpqa.p","wb"))

## NER and POS maker############################################################################

########################3[RUN THIS]############################3
def rem_mentions_hasht(tweet):
    words = tweet.split()
    relevant_tokens = [w for w in words if '@' not in w and '#' not in w]
    return( " ".join(relevant_tokens))

##POS Features
import pickle
train_stfrd_POStagged = pickle.load( open( "train_stfrd_POStagged.p", "rb" ) )
test_stfrd_POStagged = pickle.load( open( "test_stfrd_POStagged.p", "rb" ) )

##NER Features
ner_tagged_stnfrd = pickle.load( open( "train_stfrd_NERtagged.p", "rb" ) )

ner_t_tagged_stnfrd = pickle.load( open( "test_stfrd_NERtagged.p", "rb" ) )

##Argument lexicon features
train_arg_lex = pickle.load( open( "train_arg_lex.p", "rb" ) )

test_arg_lex = pickle.load( open( "test_arg_lex.p", "rb" ) )
##Subjectivity lexicon features
train_mpqa = pickle.load( open( "train_mpqa.p", "rb" ) )
test_mpqa = pickle.load( open( "test_mpqa.p", "rb" ) )

##sentiment and subjectivity features as given by textblob
from textblob import TextBlob

train_senti = train.Tweet.apply(lambda a: TextBlob(rem_mentions_hasht(a)).sentiment[0]).reshape(train.shape[0],1)
test_senti = test.Tweet.apply(lambda a: TextBlob(rem_mentions_hasht(a)).sentiment[0]).reshape(test.shape[0],1)
train_subj = train.Tweet.apply(lambda a: TextBlob(rem_mentions_hasht(a)).sentiment[1]).reshape(train.shape[0],1)
test_subj = test.Tweet.apply(lambda a: TextBlob(rem_mentions_hasht(a)).sentiment[1]).reshape(test.shape[0],1)

##Helper Functions
def ner_features(ner_tagged):
    def ner_count(ner_tagged):
        count =  {}
        count['ORGANIZATION'] = 0
        count['LOCATION'] = 0
        count['PERSON'] = 0
        count['O'] = 0
        count = [0,0,0,0]
        for word in ner_tagged:
            if word[1] == 'O':
                count[0]+=1
            elif word[1] == 'ORGANIZATION':
                count[1]+=1
            elif word[1] == 'PERSON':
                count[2]+=1
            elif word[1] == 'LOCATION':
                count[3]+=1
        return count
        
    ner_counts = ner_tagged.apply(lambda a: ner_count(a))
    ner1 = ner_counts.apply(lambda a: a[1])
    ner2 = ner_counts.apply(lambda a: a[2])
    ner3 = ner_counts.apply(lambda a: a[3])
            
    return np.array([ner1,ner2,ner3]).T

def create_dict(pos_tagged):
    pos_dict = {}
    for i in range(0,len(pos_tagged)):
        for j in pos_tagged.iloc[i]:
            if j[1] in  pos_dict.keys():
                if j[0] not in pos_dict[j[1]]:
                    pos_dict[j[1]].append(j[0])
            else:
                pos_dict[j[1]] = []
    return pos_dict

def create_dummy_cat_cols(df, cat_cols):
    cat_dummy_cols = {}
    for col in cat_cols:
        dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False).ix[:,:-1]
        df = pd.concat([dummies,df], axis=1)
        cat_dummy_cols[col] = dummies.columns.values
    return df, cat_dummy_cols
    
##F score calculator
def f_score(predictions, actual, class_type='AGAINST'):
    tot_class_pred = sum(predictions==class_type)
    correctly_classified = sum(((actual==class_type).values)&(predictions==class_type))
    tot_class_act = sum(((actual==class_type).values))
    if tot_class_pred == 0:
        prec = 0
    else:
        prec = float(correctly_classified/tot_class_pred)
    if tot_class_act == 0:
        recall = 1
    else:
        recall = float(correctly_classified/tot_class_act)
    if prec+recall==0:
        return 0
    else:
        return ((2*prec*recall)/(prec+recall))
##Final score
def custom_scorer(actual, predictions):
    against = f_score(predictions, actual, class_type='AGAINST')
    favor = f_score(predictions, actual, class_type='FAVOR')
    return favor+against/2


########################3[RUN THIS]############################3
## all words as features
def create_features(data, data_t):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(data.Tweet)
    features_t = vectorizer.transform(data_t.Tweet)
    return features, features_t

##using nouns verbs and adjectives
def create_features(data, data_t):
    from sklearn.feature_extraction.text import CountVectorizer
    pos_dict = create_dict(train_stfrd_POStagged.ix[data.index])

    Imp_Words = list(set(pos_dict['NN']+pos_dict['NNS']+pos_dict['JJ']+pos_dict['JJR']+pos_dict['JJS']+pos_dict['VB']+pos_dict['VBD']+pos_dict['VBG']+pos_dict['VBN']+pos_dict['VBP']+pos_dict['VBZ']))

    vectorizer = CountVectorizer(vocabulary=Imp_Words)
    features = vectorizer.fit_transform(data.Tweet)
    features_t = vectorizer.transform(data_t.Tweet)
    return features, features_t


##optimising the features
def create_features(data, data_t):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(ngram_range = (1,5), min_df=5, stop_words='english', strip_accents='ascii')
    features = vectorizer.fit_transform(data.Tweet)
    features_t = vectorizer.transform(data_t.Tweet)
    return features, features_t
    
##optimising the features and mpqa subjectivity
def create_features(data, data_t):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(ngram_range = (1,5), min_df=5, stop_words='english', strip_accents='ascii')
    features = vectorizer.fit_transform(data.Tweet)
    features_ner = ner_features(ner_tagged_stnfrd.ix[data.index])
    features_sent = create_dummy_cat_cols(data, ["Sentiment"])[0].ix[:,0:2]
    features_arg = train_arg_lex[data.index]
    features_mpqa = train_mpqa[data.index]
    features_senti = train_senti[data.index]
    features_subj = train_subj[data.index]
    features = hstack([features, features_mpqa])
    features_t = vectorizer.transform(data_t.Tweet)
    features_arg_t = test_arg_lex[data_t.index]
    features_ner_t = ner_features(ner_t_tagged_stnfrd.ix[data_t.index])
    features_mpqa_t = test_mpqa[data_t.index]
    features_senti_t = test_senti[data_t.index]
    features_subj_t = test_subj[data_t.index]
    features_sent_t = create_dummy_cat_cols(data_t, ["Sentiment"])[0].ix[:,0:2]
    features_t = hstack([features_t, features_mpqa_t])
    return features, features_t
    
##optimising the features and arguing subjectivity
def create_features(data, data_t):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(ngram_range = (1,5), min_df=5, stop_words='english', strip_accents='ascii')
    features = vectorizer.fit_transform(data.Tweet)
    features_ner = ner_features(ner_tagged_stnfrd.ix[data.index])
    features_sent = create_dummy_cat_cols(data, ["Sentiment"])[0].ix[:,0:2]
    features_arg = train_arg_lex[data.index]
    features_mpqa = train_mpqa[data.index]
    features_senti = train_senti[data.index]
    features_subj = train_subj[data.index]
    features = hstack([features, features_arg])
    features_t = vectorizer.transform(data_t.Tweet)
    features_arg_t = test_arg_lex[data_t.index]
    features_ner_t = ner_features(ner_t_tagged_stnfrd.ix[data_t.index])
    features_mpqa_t = test_mpqa[data_t.index]
    features_senti_t = test_senti[data_t.index]
    features_subj_t = test_subj[data_t.index]
    features_sent_t = create_dummy_cat_cols(data_t, ["Sentiment"])[0].ix[:,0:2]
    features_t = hstack([features_t, features_arg_t])
    return features, features_t

##All features
def create_features(data, data_t):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(ngram_range = (1,5), min_df=5, stop_words='english', strip_accents='ascii')
    features = vectorizer.fit_transform(data.Tweet)
    features_ner = ner_features(ner_tagged_stnfrd.ix[data.index])
    features_sent = create_dummy_cat_cols(data, ["Sentiment"])[0].ix[:,0:2]
    features_arg = train_arg_lex[data.index]
    features_mpqa = train_mpqa[data.index]
    features_senti = train_senti[data.index]
    features_subj = train_subj[data.index]
    features = hstack([features_ner, features, features_arg, features_mpqa, features_senti, features_subj])
    features_t = vectorizer.transform(data_t.Tweet)
    features_arg_t = test_arg_lex[data_t.index]
    features_ner_t = ner_features(ner_t_tagged_stnfrd.ix[data_t.index])
    features_mpqa_t = test_mpqa[data_t.index]
    features_senti_t = test_senti[data_t.index]
    features_subj_t = test_subj[data_t.index]
    features_sent_t = create_dummy_cat_cols(data_t, ["Sentiment"])[0].ix[:,0:2]
    features_t = hstack([features_ner_t, features_t, features_arg_t, features_mpqa_t, features_senti_t, features_subj_t])
    return features, features_t
    
total_test_pred = []
total_test_act = []
for target in train.Target.unique():
    
    data = train[train.Target==target]
    data_t = test[test.Target==target]
    
    features, features_t = create_features(data, data_t)
    

    from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
    from sklearn.svm import SVC
    clf = GradientBoostingClassifier(max_depth=10)
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']
    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring=make_scorer(custom_scorer))
    tuned_parameters = [{'subsample':[.4,.6,.8,1],
                         'max_depth':[3,6,10]}]
    
    #clf = algo_comparison(models,model_params=model_params, model_param_search_grid=gs_params)
    #clf.fit(features, data.Stance)
    req_count = float(len(data.Stance)/3)
    favor_weight =1 if round(req_count/sum(data.Stance=="FAVOR"),2)<1 else round(req_count/sum(data.Stance=="FAVOR"),2)
    against_weight = 1 if round(req_count/sum(data.Stance=="AGAINST"),2)<1 else round(req_count/sum(data.Stance=="AGAINST"),2)
    none_weight = 1
#==============================================================================
#     favor_weight = round(req_count/sum(data.Stance=="FAVOR"),2)
#     against_weight = round(req_count/sum(data.Stance=="AGAINST"),2)
#     none_weight = 1
#==============================================================================
    print((favor_weight, against_weight, none_weight))
    weights = np.array([against_weight if stance=="AGAINST" else favor_weight if "FAVOR" else none_weight for stance in data.Stance])
    #clf.fit(features, data.Stance, sample_weight=weights)
    clf = GridSearchCV(GradientBoostingClassifier(), tuned_parameters, cv=5, scoring=make_scorer(custom_scorer))
    clf.fit(features.toarray(), data.Stance)
    print(target)
#==============================================================================
#     f_score_favor = f_score(clf.predict(features.toarray()), data.Stance, class_type='FAVOR')
#     f_score_against = f_score(clf.predict(features.toarray()), data.Stance, class_type='AGAINST')
#     f_score_train = (f_score_favor+f_score_against)/2
#==============================================================================
    test_pred = clf.predict(features_t.toarray())
    total_test_pred = total_test_pred+test_pred.tolist()
    total_test_act = total_test_act+data_t.Stance.tolist()
    f_score_favor_t = f_score(test_pred, data_t.Stance, class_type='FAVOR')
    f_score_against_t = f_score(test_pred, data_t.Stance, class_type='AGAINST')
    f_score_none_t = f_score(test_pred, data_t.Stance, class_type='NONE')
    f_score_test = (f_score_favor_t + f_score_against_t)/2
    print((f_score_favor_t,f_score_against_t,f_score_none_t, f_score_test))

total_test_pred = pd.Series(total_test_pred)
total_test_act = pd.Series(total_test_act)
    
f_score_favor_t = f_score(total_test_pred, total_test_act, class_type='FAVOR')
f_score_against_t = f_score(total_test_pred, total_test_act, class_type='AGAINST')
f_score_none_t = f_score(total_test_pred, total_test_act, class_type='NONE')
f_score_test = (f_score_favor_t + f_score_against_t)/2
print((f_score_favor_t,f_score_against_t,f_score_none_t, f_score_test))


from nltk.sentiment import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
sid.polarity_scores("I am so happy.")


arguing_count("According to WHO recommendations, male circumcision should be followed by at least six weeks of abstinence.".lower())

from helper_funcs import *
train.Tweet.apply(lambda a: arguing_count(a))
