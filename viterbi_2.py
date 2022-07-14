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
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
"""
from collections import Counter
import numpy as np

def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    #https://web.stanford.edu/~jurafsky/slp3/8.pdf
    #Viterbi Algorithm adapted from Jurafsky chapter 8
    laplace = 0.0001
    tag_count = Counter()
    initial_tag_count = Counter()
    transition_tag_count = Counter() 
    emission_count = {} #tag:{word:count}

    unknown_word = '__UNKNOWN__'
    unique_words = [unknown_word] #laplace word
    word_count = Counter()
    word_last_tag = {}

    for sentence in train:
        # Pairwise tag counts
        for i in range(len(sentence)-1):
            tag1 = sentence[i][1]
            tag2 = sentence[i+1][1]

            transition_tag_count.update({(tag1, tag2) : 1})
        
        # Individual word counts, copied from baseline.py
        for word, tag in sentence:
            if word not in unique_words:
                unique_words.append(word)

            if tag not in emission_count:
                emission_count[tag] = Counter()
                
            emission_count[tag].update({word : 1})
            tag_count.update({tag: 1})
            word_last_tag[word] = tag
            word_count.update({word: 1})


    total_hapax = 0
    hapax_tag_count = Counter()
    for word in word_count:
        if word_count[word] == 1:
            total_hapax += 1
            hapax_tag_count.update({word_last_tag[word] : 1})

    initial_probs = {} # tag:prob
    for tag in tag_count:
        initial_probs[tag] = np.log( (initial_tag_count[tag]+laplace)/(1+len(train)+laplace*len(tag_count)) )

    transition_probs = {} # (tag1, tag2):prob    compares all tags to all other tags.     try: print("tag1: ", tag1, " tag2: ", tag2)
    for tag1 in tag_count: 
        for tag2 in tag_count:
            transition_probs[(tag1, tag2)] = np.log( (transition_tag_count[(tag1, tag2)]+laplace)/(tag_count[tag1]+laplace*(1+len(tag_count))) )

    TAGS = [tag for tag in tag_count]
    # tag : prob
    hapax_probs = {}
    for tag in TAGS:
        hapax_probs[tag] = (hapax_tag_count[tag]+laplace) / (total_hapax+laplace*(len(TAGS)+1))

    emission_probs = {} # tag : {word : prob}
    for tag in TAGS:
        emission_probs[tag] = {}
        for word in unique_words:
            hlaplace = hapax_probs[tag] * laplace
            emission_probs[tag][word] = np.log( (emission_count[tag][word]+hlaplace)/(tag_count[tag]+hlaplace*len(unique_words)) )

    output = []
    for sentence in test:
        viterbi = np.zeros((len(TAGS), len(sentence))) #create a path probability matrix viterbi[N,T]
        backpointer = np.zeros(viterbi.shape, dtype=int)

        for i in range(len(TAGS)): # initialize trellis first column values
            if sentence[0] in emission_probs[TAGS[i]]:
                viterbi[i][0] = initial_probs[TAGS[i]] + emission_probs[TAGS[i]][sentence[0]]
            else:
                viterbi[i][0] = initial_probs[TAGS[i]] + emission_probs[TAGS[i]][unknown_word]
        
        for col in range(1, viterbi.shape[1]): # col is word, row is tag
            curr_word = sentence[col]
            for row in range(viterbi.shape[0]): # each possible tag
                possible_paths = np.zeros(viterbi.shape[0])

                for prev_row in range(viterbi.shape[0]): # each possible previous tag
                    
                    if curr_word in emission_probs[TAGS[row]]: # prev node + transition + emission
                        possible_paths[prev_row] = viterbi[prev_row][col-1] + transition_probs[TAGS[prev_row], TAGS[row]] + emission_probs[TAGS[row]][curr_word]
                    else:
                        possible_paths[prev_row] = viterbi[prev_row][col-1] + transition_probs[TAGS[prev_row], TAGS[row]] + emission_probs[TAGS[row]][unknown_word]

                best_row = np.argmax(possible_paths)
                viterbi[row][col] = possible_paths[best_row]

                backpointer[row, col] = best_row

        overall_best_row = np.argmax(viterbi[:, -1]) # backtracking
        row = overall_best_row

        iter_output = []
        for col in reversed(range(1, viterbi.shape[1])):
            iter_output.insert(0,(sentence[col], TAGS[row]))
            row = backpointer[row, col]

        iter_output.insert(0,(sentence[0], TAGS[row]))
        output.append(iter_output)

    return output