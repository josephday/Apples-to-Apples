from bs4 import BeautifulSoup
import requests
import re

def process_noun(noun):
	noun = noun.text
	noun = noun.replace("&", "and")
	noun = noun.replace(" ", "_")
	return noun

def process_adjective(adjective):
	synonyms = adjective.next_sibling
	adjective = adjective.text
	if "&" in adjective:
		return
	else:
		pass
	#synonyms = adjective.next_sibling
	synonyms = re.findall("[a-zA-Z]*", synonyms)
	synonyms = [synonym for synonym in synonyms if synonym != '']
	return [adjective] + synonyms


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
adjectives = [process_adjective(adj) for adj in adjectives if process_adjective(adj) is not None]


