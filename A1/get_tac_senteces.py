'''
Created on Jul 14, 2015

@author: jcheung
'''

import sys, os, codecs
import sklearn
import numpy as np
import copy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import svm, linear_model, naive_bayes 
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# model parameters
n = 1
lemmatize = True
lowercase = False
remove_top_percent = 0
params = str((n,lemmatize,lowercase,remove_top_percent))

stoplist = set(stopwords.words('english'))

################################################################################
# reading data set
def read_tac(year):
    '''
    Read data set and return feature matrix X and labels y.
    
    X - (ndocs x nfeats)
    Y - (ndocs)
    '''
	# modify this according to your directory structure #done
    sub_folder = 'data/tac%s' % year
    X, Y = [], []
    
    # labels
    labels_f = 'tac%s.labels' % year
    
    fh = open(os.path.join(sub_folder, labels_f))
    for line in fh:
        docid, label = line.split()
        Y.append(int(label))
    
    # tac 10
    if year == '2010':
        template = 'tac10-%04d.txt'
        s, e = 1, 921

    if year == '2011':
        template = 'tac11-%04d.txt'
        s, e = 921, 1801

    print "Using 2010 set to generate feature vector template..."
    bag_of_words = []
    for i in xrange(1, 921):
        fname = os.path.join('data/tac2010', 'tac10-%04d.txt' % i)
        bag_of_words += extract_features(fname, n, lemmatize, lowercase, get_bag_words = True)
    if remove_top_percent != 0:
        ascending = sorted([(bag_of_words.count(item),item) for item in set(bag_of_words)])
        rmv = int((remove_top_percent*(1.0/100))*len(ascending))
        bag_of_words = ascending[:-rmv]
        bag_of_words = sorted([second for first, second in bag_of_words])
    else:
        bag_of_words = sorted(set(bag_of_words))


    print "Generating feature vectors for " + year + " dataset"
    for i in xrange(s, e):
        fname = os.path.join(sub_folder, template % i)
        X.append(extract_features(fname, n, lemmatize, lowercase, bag_words = bag_of_words))
        sys.stdout.write("\r #%d"%i)
        sys.stdout.flush()
    print ""
    
    #nfeats = 100 # TODO: you'll have to figure out how many features you need
    
    # convert indices to numpy array - I didn't use this.
    # for j, x in enumerate(X):
    #     arr = np.zeros(nfeats)
    #     for index in X[j]:
    #         arr[index] += 1.0
    #     X[j] = arr


    Y = np.array(Y)
    X = np.array(X)
    return X, Y
        

################################################################################
# feature extraction

def ispunct(some_string):
    return not any(char.isalnum() for char in some_string)

def get_tokens(s):
    '''
    Tokenize into words in sentences.
    
    Returns list of strs
    '''
    retval = []
    sents = sent_tokenize(s)
    
    for sent in sents:
        tokens = word_tokenize(sent)
        retval.extend(tokens)
    return retval

def extract_features(f, n, lemmatize, lowercase, get_bag_words = False, bag_words = None):
    '''
    Extract features from text file f into a feature vector.
    
    n: maximum length of n-grams
    lemmatize: (boolean) whether or not to lemmatize
    lowercase: (boolean) whether or not to lowercase everything
    '''
    
    s = codecs.open(f, 'r', encoding = 'utf-8').read()
    s = codecs.encode(s, 'ascii', 'ignore')
    
    tokens = get_tokens(s)
    #print tokens # This demonstrates that you are reading the tokens. You can comment it out or remove this line.
    if lowercase:
    	tokens = [token.lower() for token in tokens]
    if lemmatize:
    	wnl = WordNetLemmatizer()
    	tokens = [wnl.lemmatize(token) for token in tokens]

    if n == 1:
        indices = zip(*[tokens[i:] for i in range(1)])
    elif n == 2:
        indices = zip(*[tokens[i:] for i in range(2)])
    elif n == (1,2):
        indices = zip(*[tokens[i:] for i in range(1)]) + zip(*[tokens[i:] for i in range(2)])
    else:
        print "ERROR: BAD INPUT FOR N"
        exit()

    if get_bag_words:
    	return indices
    elif bag_words:
    	vector_template = dict.fromkeys(bag_words, 0)
    	for item in indices:
    		if item in vector_template:
    			vector_template[item] = vector_template[item] + 1
    	output_vector = [second for first,second in sorted([[k,v] for k, v in vector_template.items()])]
    	return output_vector
    else:
        print "ERROR: extract_features API needs either flag get_bag_words or a list bag_words"
        exit()

    # TODO: fill this part in
   

################################################################################

# evaluation code
def accuracy(gold, predict):
    assert len(gold) == len(predict)
    corr = 0
    for i in xrange(len(gold)):
        if int(gold[i]) == int(predict[i]):
            corr += 1
    acc = float(corr) / len(gold)
    print 'Accuracy %d / %d = %.4f' % (corr, len(gold), acc)

################################################################################

if __name__ == '__main__':

	# main driver code
    params = str((n,lemmatize,lowercase, remove_top_percent))
    if os.path.isfile('saves/' + params + 'X2010.pkl') and os.path.isfile('saves/' + params + 'X2011.pkl') and os.path.isfile('saves/' + params + 'Y2011.pkl') and os.path.isfile('saves/' + params + 'Y2010.pkl'):
        print "X and Y sets found in saves folder, loading..."
        X_2010 = joblib.load('saves/' + params + 'X2010.pkl')
        X_2011 = joblib.load('saves/' + params + 'X2011.pkl')
        Y_2010 = joblib.load('saves/' + params + 'Y2010.pkl')
        Y_2011 = joblib.load('saves/' + params + 'Y2011.pkl')
        print "Done."
    else:
        print "Saves not found for X and Y, generating..."
        X_2010, Y_2010 = read_tac('2010')
        joblib.dump(X_2010, 'saves/' + params + 'X2010.pkl')
        joblib.dump(Y_2010, 'saves/' + params + 'Y2010.pkl')
        X_2011, Y_2011 = read_tac('2011')
        joblib.dump(X_2011, 'saves/' + params + 'X2011.pkl')
        joblib.dump(Y_2011, 'saves/' + params + 'Y2011.pkl')
        print "Done."

    print "Training models..."
    # scaler = StandardScaler()
    # scaler.fit(X_2010)
    # X_2010 = scaler.transform(X_2010)
    # X_2011 = scaler.transform(X_2011)
    # X_2010 = X_2010.tolist()
    # X_2011 = X_2011.tolist()
    for i,v in enumerate(X_2010):
        for j,w in enumerate(v):
            if w > 0:
                X_2010[i][j] = 1.0
    for i,v in enumerate(X_2011):
        for j,w in enumerate(v):
            if w > 0:
                X_2011[i][j] = 1.0
    X_2010 = np.array(X_2010)
    X_2011 = np.array(X_2011)

    svm_model = svm.SVC(kernel='linear')
    nb_model = naive_bayes.BernoulliNB()
    logreg_model = linear_model.LogisticRegression()
    svm_model.fit(X_2010,Y_2010)
    print "SVM fitted"
    nb_model.fit(X_2010,Y_2010)
    print "NB fitted"
    logreg_model.fit(X_2010,Y_2010)
    print "LR fitted"
    nnet_model = MLPClassifier(hidden_layer_sizes = (len(X_2010),len(X_2010)/2))
    nnet_model.fit(X_2010,Y_2010)
    print "MLP fitted"

    print "Accuracy for Logistic Regression:"
    accuracy(Y_2010, logreg_model.predict(X_2010))
    accuracy(Y_2011, logreg_model.predict(X_2011))
    print "Accuracy for SVM:"
    accuracy(Y_2010, svm_model.predict(X_2010))
    accuracy(Y_2011, svm_model.predict(X_2011))
    print "Accuracy for NB:"
    accuracy(Y_2010, nb_model.predict(X_2010))
    accuracy(Y_2011, nb_model.predict(X_2011))
    print "Accuracy for NNet:"
    accuracy(Y_2010, nnet_model.predict(X_2010))
    accuracy(Y_2011, nnet_model.predict(X_2011))

    print "Confusion Matrix:"
    print confusion_matrix(Y_2011, logreg_model.predict(X_2011))

    # X_2010_together = np.column_stack((logreg_model.predict(X_2010),svm_model.predict(X_2010),nb_model.predict(X_2010),nnet_model.predict(X_2010)))
    # X_2011_together = np.column_stack((logreg_model.predict(X_2011),svm_model.predict(X_2011),nb_model.predict(X_2011),nnet_model.predict(X_2011)))

    # nnet_ensemble = MLPClassifier()
    # nnet_ensemble.fit(X_2010_together, Y_2010)

    # print "Accuracy for NNet ensemble:"
    # accuracy(Y_2010, nnet_ensemble.predict(X_2010_together))
    # accuracy(Y_2011, nnet_ensemble.predict(X_2011_together))

    
