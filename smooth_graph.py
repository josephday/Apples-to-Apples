import csv
import pandas as pd 
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import itertools as it
import random

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
	return selection, column.get_value(selection, 0)

def evaluate_model(model, nlist, hands, adjectives, true_answers, savefig, savedata):
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
			selection, value = make_selection(model, nlist, hands[i], adjectives[i])
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

	dmean = np.mean(discard)
	kmean = np.mean(keep)
	print("Mean score when kept is {}".format(kmean))
	print("Mean score when discarded is {}".format(dmean))

	smooth_keep = []
	for value in keep:
		smooth_keep += np.random.normal(value, 0.01, 10).tolist()

	smooth_discard = []
	for value in discard:
		smooth_discard += np.random.normal(value,0.01,10).tolist()


	print("times failed: {}".format(failed))
	print("When 1 correct, got it {} times out of {} for {} percent".format(correct_1, out_of_1, float(correct_1) / out_of_1))
	print("When 2 correct, got it {} times out of {} for {} percent".format(correct_2, out_of_2, float(correct_2) / out_of_2))
	print("When 3 correct, got it {} times out of {} for {} percent".format(correct_3, out_of_3, float(correct_3) / out_of_3))
	print("When 4 correct, got it {} times out of {} for {} percent".format(correct_4, out_of_4, float(correct_4) / out_of_4))
	plt.hist(smooth_discard, bins=100, alpha=0.5, label='discard')
	plt.hist(smooth_keep, bins=100, alpha=0.5, label='keep')
	plt.legend(loc='upper right')
	plt.savefig('smoothed.png')
	plt.gcf().clear()
	acc = float(correct) / out_of
	acc1 = float(correct_1) / out_of_1
	acc2 = float(correct_2) / out_of_2
	acc3 = float(correct_3) / out_of_3
	acc4 = float(correct_4) / out_of_4
	to_save = [savefig, acc, acc1, acc2, acc3, acc4, failed]
	#with open(savedata,'a') as resultFile:
#		wr = csv.writer(resultFile, dialect='excel')
#		wr.writerow(to_save)

def one_trial(nouns, hands, adjectives, true_answers, train_params, savedata):
	size = train_params[0]
	window = train_params[1]
	lr = train_params[2]
	min_lr = train_params[3]
	epochs = train_params[4]

	model = train(size=size, window=window, alpha=lr, min_alpha=min_lr, epochs=epochs)
	savefig = "tp_{}_{}_{}_{}_{}.png".format(size, window, lr, min_lr, epochs)
	evaluate_model(model, nouns, hands, adjectives, true_answers, savefig, savedata)

def all_trials():
	with open('nouns.csv', 'r') as f:
		reader = csv.reader(f)
		nouns = list(reader)[0]
	print(nouns[1])
	sizes = [100,200,300,500]
	windows = [5,10,15]
	initial_lr = [0.02,0.025,0.03]
	final_lr = [0.0001, 0.005]
	epochs = [1,3,5]

	a = [sizes, windows, initial_lr, final_lr, epochs]
	permutations = list(it.product(*a))
	print(permutations[0])
	for i in range(0, len(permutations)):
		one_trial(nouns, hands, adjectives, true_answers, permutations[i], "train_params_data.csv")








