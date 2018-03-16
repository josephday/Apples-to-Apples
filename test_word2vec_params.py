
import pandas as pd 
import numpy as np

base_data = pd.read_csv("1000_examples.csv", nrows=1000)
hands = base_data['hand'].values
hands =  [hand[2:-2].lower().split('\', \'') for hand in hands]
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

def train(source=sentences, size=200, window=5, alpha=0.025, min_alpha=0.0001, epochs=5):
	model=word2vec.Word2Vec(source, size=size, window=window, alpha=alpha, min_alpha = min_alpha, iter=epochs, workers=4)
	return model 

def compare(model, noun, adjective):
	try:
		noun = noun.lower()
		adjective = adjective.lower()
		if "_" in noun:
			base_nouns = noun.split("_")
			return np.max([compare(model,bn,adjective) for bn in base_nouns])
		elif "_" in adjective:
			base_adjectives = adjective.split("_")
			return np.max([compare(model,noun,adj) for adj in base_adjectives])
		else:
			return model.wv.similarity(noun, adjective)
	except:
		return 0 

def calc_score(model, noun, adjectives):
	try:
		return np.max([compare(model, noun, adj) for adj in adjectives])
	except:
		return 0 

def make_selection(model, nlist, hand=['cindy_crawford', 'bill_clinton', 'taking_a_shower', 'water', 'peanuts', 'ear_wax', 'whales'], adjectives=['inspiring', 'Motivational', 'hopeful']):
	column = pd.DataFrame([calc_score(model, noun, adjectives) for noun in nlist], index=nlist)
	column[0] = softmax(column.values)
	selection = hand[np.argmax([column.get_value(card, 0) for card in hand])]
	return selection

def evaluate_model(model, nlist, hands, adjectives, true_answers):
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
	for i in range(0, 100):
		if i % 25 == 0:
			print(i)
		try:
			selection = make_selection(model, nlist, hands[i], adjectives[i])
			if '?' not in true_answers[i]:
				winners = [hands[i][x-1] for x in [int(p) for p in true_answers[i]]]
				if selection in winners:
					correct+=1
					if len(winners) == 1:
						correct_1+=1
						
					if len(winners) == 2:
						correct_2+=1
					
					if len(winners) == 3:
						correct_3+=1
						
					if len(winners) == 4:
						correct_4+=1
						
				out_of+=1
				if len(winners) == 1:
					out_of_1+=1
						
				if len(winners) == 2:
					out_of_2+=1
					
				if len(winners) == 3:
					out_of_3+=1
						
				if len(winners) == 4:
					out_of_4+=1 
			else:
				pass
		except:
			failed+=1
	print("times failed: {}".format(failed))
	print("When 1 correct, got it {} times out of {} for {} percent".format(correct_1, out_of_1, float(correct_1) / out_of_1))
	print("When 2 correct, got it {} times out of {} for {} percent".format(correct_2, out_of_2, float(correct_2) / out_of_2))
	print("When 3 correct, got it {} times out of {} for {} percent".format(correct_3, out_of_3, float(correct_3) / out_of_3))
	print("When 4 correct, got it {} times out of {} for {} percent".format(correct_4, out_of_4, float(correct_4) / out_of_4))
	return(float(correct) / out_of)
