import csv
import random


with open('nouns.csv', 'r') as f:
    reader = csv.reader(f)
    nouns = list(reader)[0]
with open('adjectives.csv', 'r') as f:
    reader = csv.reader(f)
    adjectives = list(reader)[0]
adjectives = [adj[2:-2].split('\', \'') for adj in adjectives]

all_examples = []
for i in range(1,10000):
	this_example = {'hand':None,'target':None,'synonyms':None}
	possible_nouns = random.sample(nouns,7)
	target_adjective = random.sample(adjectives,1)
	synonyms = target_adjective[0][1:]
	target_adjective = target_adjective[0][0]
	this_example['hand'] = str(possible_nouns)
	this_example['target'] = target_adjective
	this_example['synonyms'] = synonyms
	this_example['1'] = possible_nouns[0]
	this_example['2'] = possible_nouns[1]
	this_example['3'] = possible_nouns[2]
	this_example['4'] = possible_nouns[3]
	this_example['5'] = possible_nouns[4]
	this_example['6'] = possible_nouns[5]
	this_example['7'] = possible_nouns[6]
	all_examples.append(this_example)

keys = all_examples[0].keys()
with open('examples.csv', 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(all_examples)




