from flask import Flask, render_template, request, flash
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import os
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

train_dir1 = r"C:\Users\PARTH\PycharmProjects\email-spam\enron1\ham"
train_dir2 = r"C:\Users\PARTH\PycharmProjects\email-spam\enron1\spam"

from collections import Counter
import codecs


def make_vocabulary(train_dir):
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with codecs.open(mail, 'r', encoding='utf-8', errors='ignore') as m:
            for i, line in enumerate(m):
                if i == 2:
                    words = line.split()
                    all_words += words

    dictionary = Counter(all_words)
    list_to_remove = list(dictionary.keys())
    for item in list_to_remove:
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    return dictionary

dictionary1=make_vocabulary(train_dir1)
dictionary2=make_vocabulary(train_dir2)
dictionary=dictionary2+dictionary1
dictionary = dictionary.most_common(3000)

vocab = {}
for i, (word, frequency) in enumerate(dictionary):
    vocab[word] = i

def extract_features(mail_dir1, mail_dir2, vocab):
    files1 = [os.path.join(mail_dir1, fi) for fi in os.listdir(mail_dir1)]
    files2 = [os.path.join(mail_dir2, fi) for fi in os.listdir(mail_dir2)]
    files = files1 + files2
    docId = 0
    matrix = np.zeros((len(files), 3000))
    for file in files:
        with codecs.open(file, 'r', encoding='utf-8', errors='ignore') as m:
            for i, line in enumerate(m):
                if i == 2:
                    words = line.split()
                    wordId = 0
                    for word in words:
                        if word not in vocab.keys():
                            continue
                        wordId = vocab[word]
                        matrix[docId, wordId] = words.count(word)
            docId += 1

    return matrix

words=extract_features(train_dir1,train_dir2,vocab)

def test_features(subj, email, vocab):

    matrix = np.zeros((1, 3000))
    docId = 0
    words = email.split()
    wordId = 0
    for word in words:
        if word not in vocab.keys():
            continue
        wordId = vocab[word]
        matrix[docId, wordId] = words.count(word)

    return matrix

train_labels=np.zeros(5172)
train_labels[0:3671]=1

app = Flask(__name__)
app.secret_key = "Unhackable"

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method=="POST":
        subj = request.form.get("subj")
        email = request.form.get("ta")
        print(subj, email)

        nb = MultinomialNB(alpha=0.25)
        nb.fit(words, train_labels)

        test_word_matrix = test_features(subj, email, vocab)

        nb_result = nb.predict(test_word_matrix)
        if nb_result:
            flash("Email is not spam", "success")
        else:
            flash("Email is spam", "error")

    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)