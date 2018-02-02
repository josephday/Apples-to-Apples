from gensim.corpora import WikiCorpus
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
print('BEGIN')
corpus = WikiCorpus('/Volumes/joe-8tb/enwiki-20180120-pages-articles.xml.bz2', dictionary=False)
max_sentence = -1
print('YO')
def generate_lines():
	for index, text in enumerate(corpus.get_texts()):
		if index < max_sentence or max_sentence==-1:
			yield text
		else:
			break

from gensim.models.word2vec import BrownCorpus, Word2Vec
model = Word2Vec()
model.build_vocab(generate_lines())
print('SUP')
model.train(generate_lines(), chunksize=500)