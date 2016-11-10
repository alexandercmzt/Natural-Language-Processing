import argparse
import sys, os, re
import nltk
from nltk.tag import hmm
from nltk.probability import LaplaceProbDist

#this is for parsing arguments.
parser = argparse.ArgumentParser(description="Deciphering")
parser.add_argument('-laplace', dest='laplace', action='store_true')
parser.add_argument('-lm', dest='lm', action='store_true')
parser.add_argument(dest='folder', type=str, default = None)
parser.set_defaults(laplace=False, lm=False)
args = parser.parse_args()

#this loads the training data into lists.
train_cipher = open(args.folder + "/train_cipher.txt")
train_plain = open(args.folder + "/train_plain.txt")
train_letters = re.split(r'[\n\r]+' , train_cipher.read())
train_cipher.close()
train_tags = re.split(r'[\n\r]+' , train_plain.read())
train_plain.close()
train_data = []
if len(train_letters) != len(train_tags):
	print "Failure: training letters and training tags are different length. Bad input files."
	exit()
else:
	for i in xrange(len(train_letters)):
		tmp = []
		for j in xrange(len(train_letters[i])):
			tmp.append((train_letters[i][j], train_tags[i][j]))
		train_data.append(tmp)
if args.lm:
	#this loads the language model as hidden state transitions.
	tac_file = open(args.folder + "/../tac-data.txt")
	tac_sentences = re.split(r'[\n\r]+' , tac_file.read())
	tac_file.close()
	for i in xrange(len(tac_sentences)):
		tmp = []
		for j in xrange(len(tac_sentences[i])):
			#we are only adding the state transitions, no emissions in this case
			tmp.append((None,tac_sentences[i][j]))
		train_data.append(tmp)

#this loads the testing data into lists
test_cipher = open(args.folder + "/test_cipher.txt")
test_plain = open(args.folder + "/test_plain.txt")
test_letters = re.split(r'[\n\r]+' , test_cipher.read())
test_cipher.close()
test_tags = re.split(r'[\n\r]+' , test_plain.read())
test_plain.close()
test_data = []
for i in xrange(len(test_letters)):
		tmp = []
		for j in xrange(len(test_letters[i])):
			tmp.append((test_letters[i][j], test_tags[i][j]))
		test_data.append(tmp)


#this trains the hmm
if args.laplace:
	trainer = hmm.HiddenMarkovModelTrainer()
	tagger = trainer.train_supervised(train_data, LaplaceProbDist)
else:
	trainer = hmm.HiddenMarkovModelTrainer()
	tagger = trainer.train_supervised(train_data)


#this gets the output and score
output_lines = []
gold_tags = []
for i in xrange(len(test_letters)):
	t = [x[1] for x in tagger.tag(test_letters[i])]
	print ''.join(t)
	output_lines += t
	gold_tags += list(test_tags[i])
	#print ''.join(list(test_tags[i]))

def accuracy(gold, predict):
    assert len(gold) == len(predict)
    corr = 0
    for i in xrange(len(gold)):
        if gold[i] == predict[i]:
            corr += 1
    acc = float(corr) / len(gold)
    print 'Accuracy %d / %d = %.4f' % (corr, len(gold), acc)


accuracy(gold_tags, output_lines)