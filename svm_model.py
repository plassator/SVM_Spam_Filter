import pandas as pd
import re
import pickle
from pathlib import Path
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def print_the_learning_curve(train_acc,test_acc,x):    
    fig,ax = plt.subplots()
    
    ax.plot(x,train_acc,label='train')
    ax.plot(x,test_acc,label='test')
    ax.set_xlabel('C')
    ax.set_ylabel('loss')
    ax.set_ylim(.0,1)
    ax.set_title(f'Learning curve for C')
    plt.legend(loc='upper right')
    plt.savefig('./SVC/C/svc_C_learning-curve.png')


def read_in_data():

    data = {'subject':[],'content':[],'spam':[]}
    p = Path("../data")

    for folder in [x for x in p.iterdir() if x.is_dir()]:
            for file in folder.glob('*.txt'):
                    with open(file) as f:
                            try:
                                subject = f.readline()
                                data['subject'].append(subject)

                                #if file ends after subject
                                try:
                                    lines = f.readlines()
                                    if len(lines)>=1: data['content'].append("".join(l for l in lines))
                                    else: data['content'].append("null")

                                    if 'spam' in file.name:
                                        spam = 1
                                    else:
                                        spam = 0
                                    data['spam'].append(spam)
                                except:
                                    data['content'].append("null")
                                    if 'spam' in file.name:
                                        spam = 0
                                    else:
                                        spam = 1
                                    data['spam'].append(spam)
                            except:
                                # loose ca 10 mails
                                continue

    df = pd.DataFrame(data)
    return df

def get_frequent_words(df):
    words = {}

    exceptions = ['the', 'in', 'Sie', '\u200c', '&zwnj;', 'und', 'der', 'die', 'to', 'of', 'and', 'a', 'den', 'von', 'mit', 'zu', '-', '(', 'auf', 'sich', 'für', 'an', 'for', 'nicht', 'ist', 'das', 'oder', 'im', 'eine', 'is', 'was', 'des', 'on', '|', 'with', 'that', '&', 'fur', 'ein', 'dem', 'this', 'Die', 'Jetzt', 'bei', 'The', 'you', 'his', 'als', 'from', '0']

    def words_dictionary(mail_body):
        for word in mail_body.split():
            #extract all words
            if word not in exceptions and len(word)>1:
                # replace all operants in the strings
                word = word.replace('(','').replace(')','').replace('[','').replace('[','').replace('+','').replace('-','').replace('*','').replace('?','').replace('\\','')
            #check for 'zeroed' words
            if len(word) > 1:
                value = words.setdefault(word,0)
                words[word] = value + 1

    for mail_body in df['content']:
        words_dictionary(mail_body)

    words_list = sorted(words,key=lambda word:words[word],reverse=True)
    
    # take the 1500 most valuable
    return words_list[:1500]


def make_matrix(df,words_list):

    matrix = []

    i = 0
    for mail_body in df['content']:
        if i%1000 == 0:
            print(len(df['content'])-i)
        vektor = []

        for word in words_list:
            if len(word)>1:
                matches=re.findall(word,mail_body)
                if matches:
                    vektor.append(len(matches))
                else:
                    vektor.append(0)
        matrix.append(vektor)
        i+=1

    pickle.dump(matrix,open("svc-features.txt","wb"))   

    return matrix

def train_the_model(matrix,df):

    X_train, X_test, y_train, y_test = train_test_split(matrix,df['spam'],test_size=.2,train_size=.8)



    """
    # increase C to fit the data better
    # gamma = auto => 1/n_features
    # gamme = scale => 1/n_features*X.var()
    train_acc,test_acc,x = [],[],[]
    #for i in range(.5,4,.5):
    c = 1
    while c < 15:
        svc_model = svm.SVC(C=c,gamma="auto")
        svc_model.fit(X_train,y_train)

        test_accuracy = svc_model.score(X_test,y_test)
        train_accuracy = svc_model.score(X_train,y_train)
        print(f"--------- C = {c} ---------")
        print(f'test accuracy: {test_accuracy}\ntrain accuracy: {train_accuracy}')
        test_acc.append(test_accuracy)
        train_acc.append(train_accuracy)

        x.append(c)
        pickle.dump(svc_model,open(f"SVC/C/svc-C-{c}_g-auto.txt","wb"))
        c += 1

    print_the_learning_curve(train_acc,test_acc,x)

    # load saved model
    #svc_model = pickle.load(open("svc-mail-model.txt","rb"))
    """

    svc_model = svm.SVC(C=9.5,gamma="auto")
    svc_model.fit(X_train,y_train)
    test_accuracy = svc_model.score(X_test,y_test)
    train_accuracy = svc_model.score(X_train,y_train)
    print(f"--------- C = {9.5} ---------")
    print(f'test accuracy: {test_accuracy}\ntrain accuracy: {train_accuracy}')
    

if __name__ == "__main__":
    df = read_in_data()
    #words_list = get_frequent_words(df)
    #matrix = make_matrix(df,words_list)
    matrix = pickle.load(open("svc-features.txt","rb"))
    train_the_model(matrix,df)

    """
    Results:
    C=1.5,gamma=auto,kernel=rbf test accuracy = 85,6%

    best test-accuracy so far:
    C=9.5,gamma=auto,kernel=rbf 1500 words accuracy = 86,93%
    """
