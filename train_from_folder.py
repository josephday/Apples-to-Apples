#train from folder

import os

class WordTrainer(object):
   def __init__(self, dir_name):
      self.dir_name = dir_name
   def __iter__(self):
      for idx,file_name in enumerate(os.listdir(self.dir_name)):
        for idxx,line in enumerate(open(os.path.join(self.dir_name, file_name),'r'):
            words = [word.lower() for word in line.split()]
            yield words

 patient_details = WordTrainer('/path/for/records/folder')
 word_vector_model = gensim.models.Word2Vec(patient_details, size=100, window=8, min_count=5)