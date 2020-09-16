from flask import Flask, redirect, url_for, render_template, request, session, flash, jsonify
from datetime import timedelta
import pickle
import numpy as np
from flask_sqlalchemy import SQLAlchemy

#model = pickle.load(open('iri.pkl', 'rb' ))
from torch_utils import transform_image, get_predictions

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

lemmatizer = nltk.WordNetLemmatizer()

# Tokenisation, Normalisation, Capitalisation, Non-alphanumeric removal, Stemming-Lemmatization
def preprocess(string):
    # to lowercase, non-alphanumeric removal
    step1 = " ".join(re.findall(r'\w+', string, flags=re.UNICODE)).lower()
    step2 = [lemmatizer.lemmatize(t).lower() for t in nltk.word_tokenize(step1)]

    return step2


def extract_tfidf(test_headlines="", test_bodies=""):
    # Body vectorisation
    body_vectorizer = pickle.load(open('body_vectorizer.pk', 'rb'))
    headline_vectorizer = pickle.load(open('headline_vectorizer.pk', 'rb'))


    print(test_bodies, test_headlines)
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

    test_tfidf = scipy.sparse.hstack([bodies_tfidf_test, headlines_tfidf_test])

    return test_tfidf



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



def combine_features(tfidf_vectors, cosine_similarity, word_overlap):
    combined_features =  sparse.bmat([[tfidf_vectors, word_overlap.T, cosine_similarity.T]])
    return combined_features


def predict(self, text):
        predicted = self.logR_pipeline.predict([text])
        predicedProb = self.logR_pipeline.predict_proba([text])[:,1]
        return bool(predicted), float(predicedProb)


lr = pickle.load(open('stance.pkl', 'rb'))


app = Flask(__name__)
app.secret_key = "kimi"
app.permanent_session_lifetime = timedelta(days=5) #minutes or hours 
app.config['SQLAlCHEMY_DATABASE_URI'] = 'sqlite:///users.sqlite3 '
app.config["SQLAlCHEMY_TRACK_MODIFICATIONS"] = False

db =SQLAlchemy(app)

class users(db.Model):
    _id = db.Column("id", db.Integer, primary_key=True)
    name = db.Column("name", db.String(100))
    email = db.Column("email", db.String(100))
    def __init__(self, name, email):
        self.name = name
        self.email = email

@app.route("/view")
def view():
    return render_template("view.html", values = users.query.all())


@app.route("/")
def test():
    return render_template("index.html")

@app.route("/home")
def home():
    return render_template("home.html")
   

@app.route("/stance")
def stance():
    return render_template("stance.html")

@app.route('/predictstance', methods=['POST'])
def predictstance():
    print('entering the predict function...')
    test_headlines = request.form['a']
    test_bodies = request.form['b']
    print('a', test_bodies)
    print('b', test_headlines)
    test_bodies = [test_bodies]
    test_headlines = [test_headlines]
    test_tfidf = extract_tfidf(test_headlines, test_bodies)
    test_overlap = extract_word_overlap(test_headlines, test_bodies)
    test_cos = extract_cosine_similarity(test_headlines, test_bodies)
    test_features = combine_features(test_tfidf, test_cos, test_overlap)
    y_pred_lr = lr.predict(test_features)
    print(y_pred_lr)
    return render_template('after_stance.html', data=y_pred_lr)


@app.route('/iris')
def iris():
    return render_template('iris.html')

@app.route('/predictiris', methods=['POST'])
def predictiris():
    print('entering the predict function...')
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)
    return render_template('after_iris.html', data=pred)

ALLOWED_EXTENSION ={'png', 'jpg', 'jpeg'}
def allowed_file(filename):
   #xx.png
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSION


@app.route('/testpredict' , methods =["POST"])
def testpredict():
    if request.method =="POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error" : "no file"})
        if not allowed_file(file.filename):
            return jsonify({"error":"format not supported"}) 

        img_bytes = file.read()
        tensor = transform_image(img_bytes)
        prediction = get_predictions(tensor)
        data = {'prediction': prediction.item(), 'class_name' :str(prediction.item())}
        return jsonify(data)

        # try:
        #     img_bytes = file.read()
        #     tensor = transform_image(img_bytes)
        #     prediction = get_predictions(tensor)
        #     data = {'prediction': prediction.item(), 'class_name' :str(prediction.item())}
        #     return jsonify(data)
        # except:
        #     return jsonify({"error":"error during prediction"})  
 
    return jsonify({'result':1})


@app.route('/get_pred_result' , methods =["GET"])
def get_pred_result():
    # 1. load image
    # 2 image ->
    # 3. prediction
    # 4. return json
    return jsonify({'result':300, 'toxicity': 0.5, 'fake_news_score': 0.67})

@app.route("/toxicity")
def toxicity():
    return render_template("toxicity.html")

@app.route("/sentiment")
def sentiment():
    return render_template("sentiment.html")

@app.route("/game")
def game():
    return render_template("game.html")


@app.route("/login", methods=[ "POST", "GET"])
def login():
    if request.method == "POST":
        session.permanent = True   #by default its false (explicitely apply true)
        user = request.form["nm"]
        session["user"] = user 

        found_user =  users.query.filter_by(name= user).first()
        if found_user == True:
            session["email"] = found_user.email
        else:
            usr =  users(user, None)
            #save to the database
            db.session.add(usr)
            db.session.commit() # staging area  


        flash("You have logged in successfully !!")
        return redirect(url_for("user"))
    else:
        if "user" in session:
            flash("You are still logged in!!" )
            return redirect(url_for("user"))
        return render_template("login.html")


@app.route("/user" ,  methods=[ "POST", "GET"])
def user():
    email = None
    if "user" in session:
        user = session["user"]
        if request.method == "POST":
            email = request.form["email"]
            session["email"] = email 
            found_user =  users.query.filter_by(name=user).first()
            found_user.email = email 
            db.session.commit()
            flash("Email was saved!!")
        else:
            if "email" in session:
                email = session["email"]
                

        return render_template("user.html", email=email)
    else:
        flash("You are not logged in !!")
        return redirect(url_for("login"))

@app.route("/logout")
def logout():
    flash("You have been logged out!!", "info") #or error
    session.pop("user", None) 
    session.pop("email", None) 
    return redirect(url_for("login"))

# @app.route("/<usr>")
# def user(usr):
#     return " welcome " +usr

# @app.route("/admin")
# def admin():
#     return redirect(url_for("test", name="Admin"))

if __name__ == "__main__":
    db.create_all()
    app.run(debug=True)