import csv
import pandas as pd 
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import itertools as it

def process_hand(hand):
	hand = hand.replace("\'", "")
	hand = hand.replace("\"", "")
	hand = hand[1:-1].lower().split(', ')
	return hand



base_data = pd.read_csv("1000_examples.csv", nrows=1000)
hands = base_data['hand'].values
hands = [process_hand(hand) for hand in hands]
synonyms = base_data['synonyms'].values
synonyms =  [hand[2:-2].lower().split('\', \'') for hand in synonyms]
adjectives = base_data['TARGET ADJECTIVE'].values
adjectives = [[adjective.lower()] for adjective in adjectives]
adjectives = [m+n for m,n in zip(adjectives,synonyms)]
true_answers = base_data['ANSWERS'].values
true_answers = [list(answer) for answer in true_answers]

from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus('text8')


def softmax(x):
	e_x = np.exp(x-np.max(x))
	return e_x / e_x.sum() * 1000

def train(source=sentences, size=500, window=15, alpha=0.03, min_alpha=0.005, epochs=5):
	model=word2vec.Word2Vec(source, size=size, window=window, alpha=alpha, min_alpha = min_alpha, iter=epochs, workers=4)
	return model 

def compare(model, noun, adjective, comp_method):
	if comp_method == 'maximum':	
		try:
			noun = noun.lower()
			adjective = adjective.lower()
			if "_" in noun:
				base_nouns = noun.split("_")
				return np.max([compare(model,bn,adjective, comp_method) for bn in base_nouns])
			elif "_" in adjective:
				base_adjectives = adjective.split("_")
				return np.max([compare(model,noun,adj, comp_method) for adj in base_adjectives])
			else:
				return model.wv.similarity(noun, adjective)
		except:
			return 0 
	elif comp_method == 'average':
		try:
			noun = noun.lower()
			adjective = adjective.lower()
			if "_" in noun:
				base_nouns = noun.split("_")
				return np.mean([compare(model,bn,adjective,comp_method) for bn in base_nouns])
			elif "_" in adjective:
				base_adjectives = adjective.split("_")
				return np.mean([compare(model,noun,adj,comp_method) for adj in base_adjectives])
			else:
				return model.wv.similarity(noun, adjective)
		except:
			return 0 
	else:
		#phrased
		try:
			noun = noun.lower()
			adjective = adjective.lower()
			if "_" in adjective:
				base_adjectives = adjective.split("_")
				return np.mean([compare(model,noun,adj,comp_method) for adj in base_adjectives])
			else:
				return model.wv.similarity(noun, adjective)
		except:
			return 0 


def calc_score(model, noun, adjectives, synonyms='maximum', comp_method='average'):
	if synonyms == 'maximum':	
		try:
			return np.max([compare(model, noun, adj, comp_method) for adj in adjectives])
		except:
			return 0 
	elif synonyms == 'average':	
		try:
			return np.mean([compare(model, noun, adj, comp_method) for adj in adjectives])
		except:
			return 0

	else:
		try:
			return compare(model, noun, adjectives[0], comp_method)
		except:
			return 0 

def make_selection(model, nlist, hand=['cindy_crawford', 'bill_clinton', 'taking_a_shower', 'water', 'peanuts', 'ear_wax', 'whales'],
					 adjectives=['inspiring', 'Motivational', 'hopeful'], synonyms='none', comp_method='none'):
	column = pd.DataFrame([calc_score(model, noun, adjectives, synonyms, comp_method) for noun in nlist], index=nlist)
	column[0] = softmax(column.values)
	selection = hand[np.argmax([calc_score(model,card,[adjectives[0]]) for card in hand])]
	return selection#, column.get_value(selection, 0)

def evaluate_model(model, nlist, hands, adjectives, true_answers, synonyms='maximum', comp_method='average'):
	correct_1 = 0
	correct_2 = 0
	correct_3 = 0
	correct_4 = 0
	out_of_1 = 0.00001
	out_of_2 = 0.00001
	out_of_3 = 0.00001
	out_of_4 = 0.00001
	correct =  0
	out_of = 0.00001
	failed= 0 
	discard = []
	keep = []
	for i in range(0, len(hands)):
		if i % 250 == 0:
			print(i)
		try:
			selection, value = make_selection(model, nlist, hands[i], adjectives[i], synonyms)
			if '?' not in true_answers[i]:
				keep.append(value)
				winners = [hands[i][x-1] for x in [int(p) for p in true_answers[i]]]
				if selection in winners:
					correct+=1
					if len(winners) == 1:
						correct_1+=1
						
					elif len(winners) == 2:
						correct_2+=1
					
					elif len(winners) == 3:
						correct_3+=1
						
					else:
						correct_4+=1
						
				out_of+=1
				if len(winners) == 1:
					out_of_1+=1
						
				elif len(winners) == 2:
					out_of_2+=1
					
				elif len(winners) == 3:
					out_of_3+=1
				else:
					out_of_4+=1 
			else:
				discard.append(value)
		except:
			print(hands[i], adjectives[i])
			failed+=1
	print("times failed: {}".format(failed))
	print("When 1 correct, got it {} times out of {} for {} percent".format(correct_1, out_of_1, float(correct_1) / out_of_1))
	print("When 2 correct, got it {} times out of {} for {} percent".format(correct_2, out_of_2, float(correct_2) / out_of_2))
	print("When 3 correct, got it {} times out of {} for {} percent".format(correct_3, out_of_3, float(correct_3) / out_of_3))
	print("When 4 correct, got it {} times out of {} for {} percent".format(correct_4, out_of_4, float(correct_4) / out_of_4))
	plt.hist(discard, bins=40, alpha=0.5, label='discard')
	plt.hist(keep, bins=40, alpha=0.5, label='keep')
	plt.legend(loc='upper right')
	#plt.savefig(savefig)
	plt.gcf().clear()
	acc = float(correct) / out_of
	print("overall: {}".format(acc))
	








