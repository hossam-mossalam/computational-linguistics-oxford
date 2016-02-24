import os as os
import math as math
import operator
import numpy as np
from random import shuffle
from collections import defaultdict

# This method reads the whole dataset
def readWSJDataset(directory = None):
  if directory == None:
    return []
  os.chdir(directory)
  data = []
  subDirectories = [d[0] for d in os.walk(os.getcwd())]
  subDirectories = subDirectories[1:]
  for d in subDirectories:
    os.chdir(d)
    for fname in os.listdir(os.getcwd()):
      f = open(fname, 'r')
      s = ''
      for line in f:
        s += line
      data += [s]
  return data

# This method returns the files with all sentences separated with the
# sequence of =s
def removeFormat(data = None):
  if data == None:
    return []
  exclude = '[]'
  data = [''.join(ch for ch in f if ch not in exclude) for f in data]
  data = [' '.join(f.split()) for f in data]
  sentenceSeparator = '======================================'
  data = [f.replace('./.', sentenceSeparator) for f in data]
  data = [f.replace('?/.', sentenceSeparator) for f in data]
  data = [f.replace('!/.', sentenceSeparator) for f in data]
  data = [f.split(sentenceSeparator) for f in data]
  data = [d for d in data if d != [''] ]
  return data

# This method is used to parse the sentences and clean them
# from all things to be ignored in the sentences
def cleanSentence(sentence):
  sentence = sentence.split(' ')
  text = []
  tags = []
  for wordTag in sentence:
    if wordTag == '': continue
    if not '/' in wordTag:
      continue
    word = wordTag[:wordTag.rfind('/')]
    tag = wordTag[wordTag.rfind('/') + 1:]
    # Unnecessary tags
    if tag in set(['``', '.', ',', "\'\'", ')', '(', ':', \
        '\n', '$', '#']):
      continue
    # When a word is tagged twice
    if '|' in tag:
      tag = tag[:tag.index('|')]
    # Removing spaces from the word and the tag
    word = (word.rstrip()).lstrip()
    tag = (tag.rstrip()).lstrip()
    text += [word]
    tags += [tag]
  return text, tags

# This method returns all the counts needed for the viterbi algorithm
def counts(data = None):
  if data == None:
    return []
  # Defining the necessary dictionaries
  wordCount = defaultdict(int)
  tagCount = defaultdict(int)
  wordTagCount = defaultdict(int)
  trigramTags = defaultdict(int)
  bigramTags = defaultdict(int)
  tagWords = defaultdict(set)
  for i, f in zip(range(len(data)), data):
    for sentence in f:
      secondPrevTag = 'START'
      prevTag = 'START'
      words, tags = cleanSentence(sentence)
      words += ['END']
      tags += ['END']
      for word, tag in zip(words, tags):
        wordCount[word] += 1
        tagCount[tag] += 1
        tagWords[tag].add(word)
        wordTagCount[word, tag] += 1
        trigramTags[secondPrevTag, prevTag, tag] += 1
        bigramTags[prevTag, tag] += 1
        secondPrevTag = prevTag
        prevTag = tag
  # Setting the tag count of START to be equal to that of END
  tagCount['START'] = tagCount['END']
  return wordCount, tagCount, wordTagCount, trigramTags, bigramTags, tagWords

# Calculate the emission probability of a word given a tag
def emission(wordCount, tagCount, wordTagCount):
  emissions = defaultdict(float)
  for word in wordCount:
    for tag in tagCount:
      emissions[word, tag] = wordTagCount[word, tag] / tagCount[tag]
  return emissions

# Calculate the trigram probability using linear interpolation for smoothing
def trigrams(tagCount, bigramTags, trigramTags, lambdaTrigrams, lambdaBigrams):
  totalTags = sum(tagCount.values())
  lambdaUnigrams = 1 - (lambdaTrigrams + lambdaBigrams)
  trigrams = defaultdict(float)
  tags = list(tagCount.keys())
  for t1 in tags + ['START']:
    for t2 in tags:
      for t3 in tags + ['END']:
        bigramCount = bigramTags[t1, t2]
        if bigramCount == 0:
          bigramCount += 1
        trigrams[t1, t2, t3] = lambdaTrigrams * \
            (trigramTags[t1, t2, t3] / bigramCount) + \
            lambdaBigrams * (bigramTags[t2, t3] / tagCount[t2]) + \
            lambdaUnigrams * (tagCount[t3] / totalTags)
  # Calculating the trigram probabilities for the start of the sentences
  for t in tags:
    trigrams['START', 'START', t] = lambdaTrigrams * \
        (trigramTags['START', 'START', t] / tagCount['START']) + \
        lambdaBigrams * (bigramTags['START', t] / tagCount['START']) + \
        lambdaUnigrams * (tagCount[t3] / totalTags)
  return trigrams

# This method was used tests if s is a number
# It was used for improving accuracy but it didn't add much accuracy
def isNumber(s):
  try:
    float(s)
    return True
  except ValueError:
    return False

# This method returns all possible tags for any index in the sentence.
# Basically it ensures that the first 2 tags have to be START
def getTagsForWordIndex(index, tags):
  if index == 0 or index == 1:
    return ['START']
  else:
    return tags

# This method is used to solve the problem of unkown words
def checkEmissions(tags, emissionProbabilitites, sentence):
  for word in sentence:
    hasEmission = False
    for tag in tags:
      if emissionProbabilitites[word, tag] > 0:
        hasEmission = True
        break
    # Smoothing the emission probability by setting all tags to have
    # equal probabilities for the unknown word
    if not hasEmission:
      probability = 1 / len(tags)
      for tag in tags:
        emissionProbabilitites[word, tag] += probability
  return emissionProbabilitites

# This is the implementation of the viterbi algorithm using the trigram model
def viterbi(emissionProbabilitites, trigramProbabilities, sentence, tags):
  score = defaultdict(float)
  bp = dict()
  emissionProbabilitites = checkEmissions(tags, emissionProbabilitites, sentence)
  sentence = ['START', 'START'] + sentence
  # Initializations
  score[1, 'START', 'START'] = 1
  score[0, 'START', 'START'] = 1
  # Viterbi Algorithm
  for k in range(2, len(sentence)):
    for u in getTagsForWordIndex(k - 1, tags):
      for v in getTagsForWordIndex(k, tags):
        optimalScore = float("-inf")
        optimalTag = 'NONE'
        for w in getTagsForWordIndex(k - 2, tags):
          if score[k - 1, w, u] * trigramProbabilities[w, u, v] * \
              emissionProbabilitites[sentence[k], v] > optimalScore:
                optimalScore = score[k - 1, w, u] * trigramProbabilities[w, u, v] * \
                    emissionProbabilitites[sentence[k], v]
                optimalTag = w
        bp[k , u, v] = optimalTag
        score[k, u, v] = optimalScore
  n = len(sentence)
  result = [0] * n
  optimalScore = float('-inf')
  # The loop used to calculate optimal accuracy
  for u in getTagsForWordIndex(n - 2, tags):
    for v in getTagsForWordIndex(n - 1, tags):
      if score[n - 1, u, v] * trigramProbabilities[u, v, 'END'] > optimalScore:
        optimalScore = score[n - 1, u, v] * trigramProbabilities[u, v, 'END']
        result[n - 2] = u
        result[n - 1] = v
  # Backtracking to get the correct tags
  for k in range(n-3, 1, -1):
    result[k] = bp[k + 2, result[k + 1], result[k + 2]]
  return result

# This method is used for evaluating accuracy
def evaluate(data, emissionProbabilitites, trigramProbabilities, tags):
  countCorrect = 0
  countWrong = 0
  size = len(data)
  i = 0
  for f in data:
    i += 1
    for sentence in f:
      if sentence == '':
        continue
      sentence, sentenceTags = cleanSentence(sentence)
      out = viterbi(emissionProbabilitites, trigramProbabilities, sentence, \
          tags)
      # Ignoring the first 2 START tags
      out = out[2:]
      for outTag, correctTag in zip(out, sentenceTags):
        if outTag == correctTag:
          countCorrect += 1
        else:
          countWrong += 1
  return countCorrect / (countCorrect + countWrong)

# This is just 90/10 test for accuracy because the trigrams take a lot of time
# to apply viterbi on
def test(data):
  print("Testing started")
  size = len(data)
  # Data was already shuffled so it is Ok to always use last 10% for testing
  testData = data[int(0.9 * size):]
  data = data[:int(0.9 * size)]
  wordCount, tagCount, wordTagCount, trigramTags, bigramTags, tagWords = \
      counts(data)
  emissionProbabilitites = emission(wordCount, tagCount, wordTagCount)
  bestAccuracy = float('-inf')
  bestLambdaTrigrams = float('inf')
  bestLambdaBigrams = float('inf')
  # Trying different values for the lambdas used in linear interpolation
  # I am starting from 0.5 but we can start from any value between 0 and 1
  # and we can end at any value between 0 and 1.
  for lambdaTrigrams in np.arange(0.5, 0.8, 0.05):
    for lambdaBigrams in np.arange(0.1, 1 - lambdaTrigrams - 0.1, 0.05):
      trigramProbabilities = trigrams(tagCount, bigramTags, trigramTags, \
          lambdaTrigrams, lambdaBigrams)
      temp = evaluate(testData, emissionProbabilitites, trigramProbabilities, \
          list(tagCount.keys()))
      # reporting current bestAccuracy
      print(lambdaTrigrams, lambdaBigrams, temp)
    if temp > bestAccuracy:
      bestAccuracy = temp
      bestLambdaTrigrams = lambdaTrigrams
      bestLambdaBigrams = lambdaBigrams
  # reporting best bestAccuracy and corresponding parameters
  print(bestLambdaTrigrams, bestLambdaBigrams, bestAccuracy)

if __name__ == '__main__':
  data = readWSJDataset('../data/')
  data = removeFormat(data)
  shuffle(data)
  test(data)
