#Joe Day
#10/26/17
#A2A Thesis Project

# Requires gensim

import nltk
from gensim.models import word2vec
from sklearn.preprocessing import normalize
#from nltk.stem import WordNetLemmatizer
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus('text8')
#nltk.download()
#lemmatiser = WordNetLemmatizer()

def train(source = sentences):
	model=word2vec.Word2Vec(sentences,size=200, window=5, min_count=5, workers=4)
	return model 

nouns = ['airplane', 'queen', 'tiger', 'dentist', 'candy', 'mirror', 'fuzz']
adjectives = ['fast', 'regal', 'dangerous', 'scary', 'sweet', 'shiny', 'fuzzy']
synonyms = [['speedy', 'quick'], ['important', 'royal'], ['unsafe', 'risky'], ['spooky', 'frightening'], ['sugary', 'nice'], ['bright', 'glossy'], ['furry', 'hairy']]

def test(nouns, adjectives, model):
	winners = []
	for adjective in adjectives:
		best_noun = None
		best_score = 0
		for noun in nouns:
			score = model.wv.similarity(noun, adjective)
			print(adjective, noun, score)			
			if score > best_score:
				best_noun = noun
				best_score = score
			else:
				pass
		winners.append((adjective, best_noun, best_score))
	return winners

def test2(nouns, adjectives, synonyms, model):
	winners = []
	for i, adjective in enumerate(adjectives):
		best_noun = None
		best_noun_score = 0
		adjective_used = adjective
		for noun in nouns:
			
			best_synonym_score = model.wv.similarity(noun, adjective)
			best_synonym = adjective
			for synonym in synonyms[i]:
				
				synonym_score = model.wv.similarity(noun, synonym)
			
				if synonym_score > best_synonym_score:
					best_synonym_score = synonym_score
					best_synonym = synonym
				else:
					pass

			if best_synonym_score > best_noun_score:
				best_noun = noun
				best_noun_score = best_synonym_score
				adjective_used = best_synonym
			else:
				pass
		winners.append((adjective, adjective_used, best_noun, best_noun_score))
	return winners

def test3(nouns, adjectives, model):
	winners = []
	for adjective in adjectives:
		lemma_a = lemmatiser.lemmatize(adjective)
		best_noun = None
		best_score = 0
		for noun in nouns:
			lemma_n = lemmatiser.lemmatize(noun)
			score = model.wv.similarity(lemma_n, lemma_a)
			
			if score > best_score:
				best_noun = noun
				best_score = score
			else:
				pass
		winners.append((adjective, best_noun, best_score))
	return winners