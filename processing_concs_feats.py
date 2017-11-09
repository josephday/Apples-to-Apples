#Joe Day

import pandas as pd 
import numpy as np

base_data = pd.read_csv("CONCS_FEATS_concstats_brm.csv", index_col=None)

from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus('text8')

def train(source = sentences):
	model=word2vec.Word2Vec(sentences,size=200, window=5, min_count=5, workers=4)
	return model 

nouns = ['genocide', 'airplane', 'queen', 'tiger', 'ghost', 'chocolate', 'mirror', 'fuzz', 'apple']
adjectives = ['tragic', 'fast', 'regal', 'dangerous', 'scary', 'sweet', 'shiny', 'fuzzy', 'red']
synonyms = [['sad', 'horrific'], ['speedy', 'quick'], ['important', 'royal'],
            ['unsafe', 'risky'], ['spooky', 'frightening'],
            ['sugary', 'nice'], ['bright', 'glossy'],
            ['furry', 'hairy'], []]


def compute_score_noun(model, noun, adjective, base_data):
	regex_adjective = '_' + adjective + '(_)?$'
	with_adjective = base_data[base_data['Feature'].str.contains(regex_adjective)].sort_values(by='Rank_PF')
	things_that_are_adjective = with_adjective['Concept'].tolist()
	if noun in things_that_are_adjective:
		freq = with_adjective[with_adjective['Concept'] == noun]['Prod_Freq'].iloc[0]
		base_score = 0.99 + freq/10000.0
	else:
		base_score = model.wv.similarity(noun, adjective)
	things_that_are_adjective = [thing for thing in things_that_are_adjective if thing in model.wv.vocab]
	if len(things_that_are_adjective) > 0:
		replacement = np.max([model.wv.similarity(noun, x) for x in things_that_are_adjective])
		if base_score < replacement:
			return round(base_score,4)
		else:
			return round(replacement,4)
	else:
		return round(base_score,4)

def one_noun_many_adjectives(model, noun, adjectives, base_data):
	scores = []
	for adjective in adjectives:
		scores.append(compute_score_noun(model, noun, adjective, base_data))
	return np.max(scores)


def find_best_noun(model, nouns, adjectives, base_data):
	scores = [one_noun_many_adjectives(model, noun, adjectives, base_data) for noun in nouns]
	index = np.argmax(scores)
	return nouns[index]


def test(model, base_data=base_data, nouns=nouns, adjectives=adjectives, synonyms=synonyms):
	for i, adjective in enumerate(adjectives):
		adj_and_syms = [adjective] + synonyms[i]
		noun_choice = find_best_noun(model, nouns, adj_and_syms, base_data)
		print(adjective, noun_choice)