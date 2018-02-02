#train from folder
import gensim
import os
import time
from multiprocessing import cpu_count

start_time = time.time()

class WordTrainer(object):
   def __init__(self, dir_name):
      self.dir_name = dir_name
      self.epoch = -1
   def __iter__(self):
      for idx,file_name in enumerate(os.listdir(self.dir_name)):
        if idx % 100 == 0:
        	print('{} files processed'.format(idx))
        if idx % 1000 == 0:
        	print('{} elapsed'.format(time.time() - start_time))
        if file_name[:4] == 'wiki':
	        for line in open(os.path.join(self.dir_name, file_name),'r'):
	            words = [word.lower() for word in line.split()]
	            yield words
      self.epoch += 1
      print('EPOCH {} COMPLETE'.format(self.epoch))
      

all_wikipedia = WordTrainer('/Volumes/joe-8tb/test')
word_vector_model = gensim.models.Word2Vec(all_wikipedia, size=200, window=5, min_count=5, workers=cpu_count())