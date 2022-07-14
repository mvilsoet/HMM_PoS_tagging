# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
from collections import Counter

def baseline(train, test):
	'''
	input:  training data (list of sentences, with tags on the words)
		test data (list of sentences, no tags on the words)
	output: list of sentences, each sentence is a list of (word,tag) pairs.
		E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
	'''

	word_tag_frequency = {}
	tag_frequency = Counter() #needed to find most common tag

	for sentence in train:
		for word, tag in sentence:
			if word not in word_tag_frequency:
				word_tag_frequency[word] = Counter() #each [word] will have a Counter() of [ (tag, frequency), ... ]

			word_tag_frequency[word].update({tag: 1}) #add one to the [word]'s correct tag
			tag_frequency.update({tag: 1})

	#print(word_tag_frequency['the'])
	most_frequent_tag = tag_frequency.most_common(1)[0][0] #most_common returns the tag with the highest total frequency

	output = []
	for sentence in test:
		
		sentence_word_tag_pairs = []
		for word in sentence:
			if word in word_tag_frequency:
				sentence_word_tag_pairs.append( (word, word_tag_frequency[word].most_common(1)[0][0]) ) #returns the tag with the highest count for a single word
			else:
				sentence_word_tag_pairs.append( (word, most_frequent_tag) ) #just guess that it's a noun lol
				
		output.append(sentence_word_tag_pairs)

	return output