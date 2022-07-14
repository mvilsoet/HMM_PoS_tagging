# HMM_PoS_tagging
Tags parts of speech input using a hidden Markov model based on Chapter 8 of [Jurafsky and Martin's "Speech and Language Processing"](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf)

The [Brown corpus](https://en.wikipedia.org/wiki/Brown_Corpus) data/brown-training.txt, data/brown-dev.txt is used in training and development.

### Tagset
16 total speech tags:

-ADJ adjective, 
-ADV adverb, 
-IN preposition, 
-PART particle (e.g. after verb, looks like a preposition), 
-PRON pronoun, 
-NUM number, 
-CONJ conjunction, 
-UH filler, exclamation, 
-TO infinitive, 
-VERB verb, 
-MODAL modal verb, 
-DET determiner, 
-NOUN noun, 
-PERIOD end of sentence punctuation, 
-PUNCT other punctuation, 
-X miscellaneous hard-to-classify items.

### Taggers

The Baseline tagger considers each word independently, ignoring previous words and tags. For each word w, it counts how many times w occurs with each tag in the training data. When processing the test data, it consistently gives w the tag that was seen most often. For unseen words, it should guess the tag that's seen the most often in training dataset.


The Viterbi tagger implements the HMM trellis (Viterbi) decoding algoirthm as seen in "Speech and Language Processing". The probability of each tag depends only on the previous tag, and the probability of each word depends only on the corresponding tag. This model estimates three sets of probabilities: Initial probabilities (How often does each tag occur at the start of a sentence?), Transition probabilities (How often does tag tb follow tag ta?), and Emission probabilities (How often does tag t yield word w?).


In viterbi_2, (out-of-vocabulary or OOV words) are considered as well in an optimization. OOV words and words that appear once in the training data (hapax words) tend to have similar parts of speech (POS). For this reason, instead of assuming that OOV words are uniformly distributed across all POS, we can get a much better estimate of their distribution by measuring the distribution of hapax words.


## Run
To run the code on the Brown corpus data, specify where the data is and which algorithm to run, either baseline, viterbi_1, viterbi_2, or viterbi_ec:

`python3 mp4.py --train data/brown-training.txt --test data/brown-dev.txt --algorithm [baseline, viterbi_1, viterbi_2]`

The optimized version of the Viterbi code has over 66.5% accuracy on the Brown development set, with an overall accuracy of about 95.5%.
