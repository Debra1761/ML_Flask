from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pylab as py
import pickle
from scipy.sparse import hstack
from scipy.sparse import coo_matrix
from tqdm import tqdm
from scipy import sparse
import csv, random, numpy, os, re, scipy, gensim
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from langdetect import detect
from sklearn.ensemble import RandomForestClassifier
from csv import DictReader
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier


# import score


import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context




nltk.download('punkt')
nltk.download('wordnet')

class DataSet():
    def __init__(self, path=""):
        self.path = path

        print("Reading dataset")
        bodies = "train_bodies.csv"
        stances = "train_stances.csv"

        self.stances = self.read(stances)
        articles = self.read(bodies)
        self.articles = dict()

        #make the body ID an integer value
        for s in self.stances:
            s['Body ID'] = int(s['Body ID'])

        #copy all bodies into a dictionary
        for article in articles:
            self.articles[int(article['Body ID'])] = article['articleBody']

        print("Total stances: " + str(len(self.stances)))
        print("Total bodies: " + str(len(self.articles)))



    def read(self,filename):
        rows = []
        with open(self.path + "" + filename, "r", encoding='utf-8') as table:
            r = DictReader(table)

            for line in r:
                rows.append(line)
        return rows


dataset = DataSet()

def read_ids(file,base):
    ids = []
    with open(base+""+file,"r") as f:
        for line in f:
            ids.append(int(line))
        return ids
    
def split(dataset, base=""):
    if not (os.path.exists(""+"training_ids.txt")
            and os.path.exists(""+"dev_ids.txt") and os.path.exists(""+"test_ids.txt")):
        raise Exception("There is an error and the dataset reader cannot find the "
                        "{training_ids|test_ids|dev_ids}.txt file. Please make sure your python paths "
                        "are configured correctly")

    training_ids = read_ids("training_ids.txt",base)
    dev_ids = read_ids("dev_ids.txt",base)
    test_ids = read_ids("test_ids.txt",base)

    #return the stances that meet these criteria
    training_stances = []
    dev_stances = []
    test_stances = []

    for stance in dataset.stances:
        if stance['Body ID'] in training_ids:
            training_stances.append(stance)
        elif stance['Body ID'] in dev_ids:
            dev_stances.append(stance)
        elif stance['Body ID'] in test_ids:
            test_stances.append(stance)


    return {"training":training_stances, "dev":dev_stances, "test": test_stances}


print("\n[1] Loading data..")
data_splits = split(dataset)
# in the format: Stance, Headline, BodyID
training_data = data_splits['training']
dev_data = data_splits['dev']
test_data = data_splits['test'] # currently 0 test points

# Change the number of training examples used.
N = int(len(training_data) * 1.0)
training_data = training_data[:N]

print("\t-Training size:\t", len(training_data))
print("\t-Dev size:\t", len(dev_data))
print("\t-Test data:\t", len(test_data))



# Get the bodies of training data points
def get_bodies(data):
    bodies = []
    for i in range(len(data)):
        bodies.append(dataset.articles[data[i]['Body ID']])
    return bodies

# Get the headlines of training data points
def get_headlines(data):
    headlines = []
    for i in range(len(data)):
        headlines.append(data[i]['Headline'])
    return headlines

# Get bodies and headlines for dev and training data
training_bodies = get_bodies(training_data)
training_headlines = get_headlines(training_data)
dev_bodies = get_bodies(dev_data)
dev_headlines = get_headlines(dev_data)
test_bodies = get_bodies(test_data)
test_headlines = get_headlines(test_data)

print(len(training_bodies))
print(len(training_headlines))
print(len(dev_bodies))
print(len(dev_headlines))
print(len(test_bodies))
print(len(test_headlines))



# Function for extracting tf-idf vectors (for both the bodies and the headlines).
# https://www.kaggle.com/adamschroeder/countvectorizer-tfidfvectorizer-predict-comments

def extract_tfidf(training_headlines, training_bodies, dev_headlines="", dev_bodies="", test_headlines="", test_bodies=""):
    # Body vectorisation
    body_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english')#, max_features=1024)
    bodies_tfidf = body_vectorizer.fit_transform(training_bodies)

    # Headline vectorisation
    headline_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english')#, max_features=1024)
    headlines_tfidf = headline_vectorizer.fit_transform(training_headlines)

    with open('body_vectorizer.pk', 'wb') as b:
        pickle.dump(body_vectorizer, b)

    with open('headline_vectorizer.pk', 'wb') as h:
        pickle.dump(headline_vectorizer, h)

    # Tranform dev/test bodies and headlines using the trained vectorizer (trained on training data)
    bodies_tfidf_dev = body_vectorizer.transform(dev_bodies)
    headlines_tfidf_dev = headline_vectorizer.transform(dev_headlines)

    bodies_tfidf_test = body_vectorizer.transform(test_bodies)
    headlines_tfidf_test = headline_vectorizer.transform(test_headlines)
    
    feature_names = np.array(body_vectorizer.get_feature_names())
    sorted_by_idf = np.argsort(body_vectorizer.idf_) 
    print('Features with lowest and highest idf in the body vector:\n')
    # The token which appears maximum times but it is also in all documents, has its idf the lowest
    print("Features with lowest idf:\n{}".format(
    feature_names[sorted_by_idf[:10]]))
    # The tokens can have the most idf weight because they are the only tokens that appear in one document only
    print("\nFeatures with highest idf:\n{}".format(
    feature_names[sorted_by_idf[-10:]]))

    # Combine body_tfdif with headline_tfidf for every data point. 
    training_tfidf = scipy.sparse.hstack([bodies_tfidf, headlines_tfidf])
    dev_tfidf = scipy.sparse.hstack([bodies_tfidf_dev, headlines_tfidf_dev])
    test_tfidf = scipy.sparse.hstack([bodies_tfidf_test, headlines_tfidf_test])

    return training_tfidf, dev_tfidf, test_tfidf



lemmatizer = nltk.WordNetLemmatizer()

# Tokenisation, Normalisation, Capitalisation, Non-alphanumeric removal, Stemming-Lemmatization
def preprocess(string):
    # to lowercase, non-alphanumeric removal
    step1 = " ".join(re.findall(r'\w+', string, flags=re.UNICODE)).lower()
    step2 = [lemmatizer.lemmatize(t).lower() for t in nltk.word_tokenize(step1)]

    return step2

# Function for extracting word overlap
def extract_word_overlap(headlines, bodies):
    word_overlap = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        preprocess_headline = preprocess(headline)
        preprocess_body = preprocess(body)
        
        # Lenght of common words b/w body and headline / Length of all the words of body & headline
        features = len(set(preprocess_headline).intersection(preprocess_body)) / float(len(set(preprocess_headline).union(preprocess_body)))
        word_overlap.append(features)
        
        # Convert the list to a sparse matrix (in order to concatenate the cos sim with other features)
        word_overlap_sparse = scipy.sparse.coo_matrix(numpy.array(word_overlap)) 
    return word_overlap_sparse


# Function for extracting the cosine similarity between bodies and headlines. 
def extract_cosine_similarity(headlines, bodies):
    vectorizer = TfidfVectorizer(ngram_range=(1,2), lowercase=True, stop_words='english')#, max_features=1024)
    
    cos_sim_features = []
    for i in range(0, len(bodies)):
        body_vs_headline = []
        body_vs_headline.append(bodies[i])
        body_vs_headline.append(headlines[i])
        tfidf = vectorizer.fit_transform(body_vs_headline)
        
        cosine_similarity = (tfidf * tfidf.T).A
        cos_sim_features.append(cosine_similarity[0][1])

    # Convert the list to a sparse matrix (in order to concatenate the cos sim with other features)
    cos_sim_array = scipy.sparse.coo_matrix(numpy.array(cos_sim_features)) 

    return cos_sim_array



# Function for combining features of various types (lists, coo_matrix, np.array etc.)
def combine_features(tfidf_vectors, cosine_similarity, word_overlap):
    combined_features =  sparse.bmat([[tfidf_vectors, word_overlap.T, cosine_similarity.T]])
    return combined_features


# Function for extracting features
# Feautres: 1) Word Overlap, 2) TF-IDF vectors, 3) Cosine similarity, 4) Word embeddings
def extract_features(train, dev, test):
# Get bodies and headlines for dev and training data
    training_bodies = get_bodies(training_data)
    training_headlines = get_headlines(training_data)
    dev_bodies = get_bodies(dev_data)
    dev_headlines = get_headlines(dev_data)
    test_bodies = get_bodies(test_data)
    test_headlines = get_headlines(test_data)

    # Extract tfidf vectors
    print("\t-Extracting tfidf vectors..")
    training_tfidf, dev_tfidf, test_tfidf = extract_tfidf(training_headlines, training_bodies, dev_headlines, dev_bodies, test_headlines, test_bodies)
    print("\t-Tfidf vectors extracted..")

    # Extract word overlap 
    print("\t-Extracting word overlap..")
    training_overlap = extract_word_overlap(training_headlines, training_bodies)
    dev_overlap = extract_word_overlap(dev_headlines, dev_bodies)
    test_overlap = extract_word_overlap(test_headlines, test_bodies)
    print("\t-Word overlap extracted..")

#     # Extract cosine similarity between bodies and headlines. 
    print("\t-Extracting cosine similarity..")
    training_cos = extract_cosine_similarity(training_headlines, training_bodies)
    dev_cos = extract_cosine_similarity(dev_headlines, dev_bodies)
    test_cos = extract_cosine_similarity(test_headlines, test_bodies)
    print("\t-Cosine similarity extracted..")

    # Combine the features
    training_features = combine_features(training_tfidf, training_cos, training_overlap)
    dev_features = combine_features(dev_tfidf, dev_cos, dev_overlap)
    test_features = combine_features(test_tfidf, test_cos, test_overlap)
    print("\t-Combined features returned..")

    return training_features, dev_features, test_features

# Feature extraction
print("[2] Extracting features.. ")
# extract_features(training_data, dev_data, test_data)
training_features, dev_features, test_features = extract_features(training_data, dev_data, test_data)


classes = ['agree', 'disagree', 'discuss', 'unrelated']

def report_score(test,pred,algo):
    #accuracy calculation
    accuracy = accuracy_score(test,pred)
    print('\n Accuracy_score for %s = %s \n'%(algo,accuracy))
    
    #confusion_matrix
    mat = confusion_matrix(test,pred)
    fig, ax = plt.subplots(figsize=(11,11))  
    ax.set_title("Confusion Matrix for %s" %algo)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=True,cmap="icefire",
                xticklabels=classes,yticklabels=classes,linewidths=.3, ax=ax)
    plt.xlabel('True label')
    plt.ylabel('Predicted label');

    #classification report
    cls = classification_report(test,pred, target_names=classes)
    print(cls)

# Creating targets
targets_tr = [a['Stance'] for a in training_data]
targets_dev = [a['Stance'] for a in dev_data]
targets_test = [a['Stance'] for a in test_data]


def change(f):
    if f == 'unrelated':
        return 0
    elif f == 'disagree':
        return 1
    elif f == 'discuss':
        return 2
    elif f == 'agree':
        return 3
    else:
        return 0

y = [change(x) for x in targets_tr]
y_test = [change(x) for x in targets_test]


def predict(self, text):
        predicted = self.logR_pipeline.predict([text])
        predicedProb = self.logR_pipeline.predict_proba([text])[:,1]
        return bool(predicted), float(predicedProb)


# Fitting model
print("[3] Fitting model..")
print("\t-Logistic Regression")
lr = LogisticRegression(C = 1.0, class_weight='balanced', solver="lbfgs", max_iter=150) 
y_pred_lr = lr.fit(training_features, targets_tr).predict(test_features)
# Evaluation
print("[4] Evaluating model..")
report_score(targets_test, y_pred_lr,'Logistic Regression')
print("\t-Done with Logistic Regression")



pickle.dump(lr, open('stance.pkl', 'wb'))
print('successfully saved model file')