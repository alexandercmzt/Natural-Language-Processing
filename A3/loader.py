'''
Created on Oct 26, 2015

@author: jcheung
'''
import xml.etree.cElementTree as ET
import codecs
import string
import re
from nltk.corpus import wordnet
from nltk.wsd import lesk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

class WSDInstance:
    def __init__(self, my_id, lemma, context, index):
        self.id = my_id         # id of the WSD instance
        self.lemma = lemma      # lemma of the word whose sense is to be resolved
        self.context = context  # lemma of all the words in the sentential context
        self.index = index      # index of lemma within the context
    def __str__(self):
        '''
        For printing purposes.
        '''
        return '%s\t%s\t%s\t%d' % (self.id, self.lemma, ' '.join(self.context), self.index)

def load_instances(f):
    '''
    Load two lists of cases to perform WSD on. The structure that is returned is a dict, where
    the keys are the ids, and the values are instances of WSDInstance.
    '''
    tree = ET.parse(f)
    root = tree.getroot()
    
    dev_instances = {}
    test_instances = {}
    
    for text in root:
        if text.attrib['id'].startswith('d001'):
            instances = dev_instances
        else:
            instances = test_instances
        for sentence in text:
            # construct sentence context
            context = [to_ascii(el.attrib['lemma']) for el in sentence]
            for i, el in enumerate(sentence):
                if el.tag == 'instance':
                    my_id = el.attrib['id']
                    lemma = to_ascii(el.attrib['lemma'])
                    instances[my_id] = WSDInstance(my_id, lemma, context, i)
    return dev_instances, test_instances

def load_key(f):
    '''
    Load the solutions as dicts.
    Key is the id
    Value is the list of correct sense keys. 
    '''
    dev_key = {}
    test_key = {}
    for line in open(f):
        if len(line) <= 1: continue
        #print line
        doc, my_id, sense_key = line.strip().split(' ', 2)
        if doc == 'd001':
            dev_key[my_id] = sense_key.split()
        else:
            test_key[my_id] = sense_key.split()
    return dev_key, test_key

def to_ascii(s):
    # remove all non-ascii characters
    return codecs.encode(s, 'ascii', 'ignore')

def first_senses(instances, how_many=1):
	#returns the most common senses from wordnet
	response = {}
	for key in instances.iterkeys():
		q = instances[key]
		a = wordnet.synsets(q.lemma)[0]
		response[key] = a.lemmas()[:how_many-1].key()
	return response

def lesk_senses(instances, window_size = None):
	#returns lesk's output
	stops = set(stopwords.words('english'))
	response = {}
	for key in instances.iterkeys():
		q = instances[key]
		new_context = []
		for elem in q.context:
			ctx = elem.translate(None, string.punctuation) #removes random punctuation throughout the data (this does not break the underscore thing mentioned in the assignment)
			if ctx != "" and ctx not in stops: #don't use stopwords in context
				new_context.append(ctx)
		if not window_size:
			response[key] = lesk(new_context, q.lemma).lemmas()[0].key()
		else:
			response[key] = lesk(new_context[q.index-window_size:q.index+window_size], q.lemma).lemmas()[0].key()
	return response

# def combined_selsnses(instances, window_size=None):
# 	fi

def accuracy(gold, pred):
	count = 0.0
	count_keyerr = 0
	for key in pred.iterkeys():
		if pred[key] in gold[key]:
			count += 1
	return count/len(gold)


if __name__ == '__main__':
	data_f = 'multilingual-all-words.en.xml'
	key_f = 'wordnet.en.key'
	dev_instances, test_instances = load_instances(data_f)
	dev_key, test_key = load_key(key_f)

	# IMPORTANT: keys contain fewer entries than the instances; need to remove them
	dev_instances = {k:v for (k,v) in dev_instances.iteritems() if k in dev_key}
	test_instances = {k:v for (k,v) in test_instances.iteritems() if k in test_key}

	sense_baseline = first_senses(test_instances)
	lesk_baseline = lesk_senses(test_instances, window_size = 2)

	print accuracy(test_key,sense_baseline)
	print accuracy(test_key,lesk_baseline)










    
    