from bs4 import BeautifulSoup
import requests
import re
import csv

def process_noun(noun):
	noun = noun.text
	noun = noun.replace("&", "and")
	noun = noun.replace(" ", "_")
	noun = noun.replace("\'", "")
	return noun.lower()

def process_adjective(adjective):
	synonyms = adjective.next_sibling
	#print(synonyms)
	adjective = adjective.text
	adjective = adjective.replace(" ", "_")
	if "&" in adjective:
		return
	else:
		pass
	#synonyms = adjective.next_sibling
	#synonyms = [synonym.replace(' ', '_') for synonym in synonyms]
	synonyms = re.findall("[a-zA-Z_\-.\']*", synonyms)
	forbidden = ['','-', 'Party', 'Set', 'set', 'party']
	synonyms = [synonym.lower() for synonym in synonyms if synonym not in forbidden]
	return [adjective.lower()] + synonyms


nouns_url = "www.com-www.com/applestoapples/applestoapples-red-without.html"
nouns_r  = requests.get("http://" + nouns_url)
nouns_data = nouns_r.text
noun_soup = BeautifulSoup(nouns_data, "html5lib")
nouns = noun_soup.find_all('b')
nouns = list(set([process_noun(noun) for noun in nouns]))

adj_url = "http://www.com-www.com/applestoapples/applestoapples-green-with.html"
adj_r = requests.get(adj_url)
adj_data = adj_r.text
adj_soup = BeautifulSoup(adj_data, "html5lib") 
adjectives = adj_soup.find_all('b')
adjectives = list(set(adjectives))
adjectives = [process_adjective(adj) for adj in adjectives if process_adjective(adj) is not None]

with open("nouns.csv",'w') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerow(nouns)

with open("adjectives.csv",'w') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerow(adjectives)

