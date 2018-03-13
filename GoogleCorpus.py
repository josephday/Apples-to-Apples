import gensim 

#Check to see if you are running 64-bit
#import struct
#print(struct.calcsize("P") * 8)

#Load Google's pre-trained word2vec
model = gensim.models.KeyedVectors.load_word2vec_format('/Volumes/joe-8tb/GoogleNews-vectors-negative300.bin', binary=True)