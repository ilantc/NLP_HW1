import math
from scipy.optimize import fmin_l_bfgs_b
import scipy
import numpy
import feature
import sentence
import time
import cPickle as pickle

class MEMMModel:
    
    def __init__(self,verbose,minFeatureCount):
        # dictionary is a mapping from word to (tag,count) tuple list
        self.dictionary = {}
        # an ordered list of functions from (sentence,index,tag_i-1.tag_i-2) to bool
        self.featureSet = []
        self.featureNum = 0
        
        # mapping from a tag to a list of feature indices that involve this tag
        # used to save calculations of features that we know will be "0" 
        self.tagToFeatureIndices = {}
        
        # mapping from tag to tag bigram features
        self.tagToTagNgramFeatureIndices = {}
        
        # list of sentence objects (for training)
        self.allSentences = []
        self.sentenceNum = 0
        
        # all possible tags
        self.tagSet = []
        
        # lamda for calculating L(v) and grad(L(V))
        self.lamda = 0
        
        # feature vector for all words with their golden tags - for saving calculations
        self.allWordsFeatureVecs = []
        
        # flag for printing
        self.verbose = verbose
        
        # minimum number of feature appearance in training data 
        # if a feature occures less than this number - dont include it in the model
        self.minFeatureCount = minFeatureCount
    
    def summarize(self):
        """ print a summary of the model""" 
        print "MODEL SUMMARY:"
        print   "\tnum sentences             =", self.sentenceNum, \
              "\n\tnum features              =", self.featureNum, \
              "\n\tnum tags                  =", len(self.tagSet), \
              "\n\tlamda                     =", self.lamda, \
              "\n\tnum words                 =", len(self.allWordsFeatureVecs), \
              "\n\tminFeatureCount =", self.minFeatureCount
    
    def show(self):
        """ heavy printing - use only for debugging of very small models"""
        print "sentences are:"
        for sentence in self.allSentences:
            print sentence.toString()
        print self.dictionary
        for f in self.featureSet:
            print f.name
    
    def reset(self):
        """ not realy needed - but here anyway for the good order of things"""
        self.dictionary = {}
        self.featureSet = []
        self.featureNum = 0
        self.tagToFeatureIndices = {}
        self.tagToTagNgramFeatureIndices = {}
        self.allSentences = []
        self.sentenceNum = 0
        self.tagSet = []
        self.lamda = 0
        self.allWordsFeatureVecs = []
        self.v = 0
    
    def saveFeaturesRaw(self):
        """ save features as raw data, since pickle.dump can't dump functions"""
        self.rawFeatures = []
        for feature in self.featureSet:
            self.rawFeatures.append(feature.toRawObj())
        self.featureSet = []
    
    def save(self,fileName):
        """ save the model to file"""
        self.saveFeaturesRaw()
        with open(fileName, 'wb') as output:
            pickler = pickle.Pickler(output, -1)
            pickler.dump(self)
    
    def loadFeaturesFromRaw(self):
        """ convert all raw feature objects to actual feature objects when loading a model from file"""
        self.featureSet = []
        for rawFeature in self.rawFeatures:
            # unfortunatly there is no switch statement in Python :( 
            if rawFeature['type'] == 'unigramWordTagFeature':
                f = feature.unigramWordTagFeature(rawFeature['word'], rawFeature['tag'], rawFeature['name'])
                self.featureSet.append(f)
                continue
            if rawFeature['type'] == 'bigramWordTagFeature':
                f = feature.bigramWordTagFeature(rawFeature['word'], rawFeature['tag'], rawFeature['prevTag'], rawFeature['name'])
                self.featureSet.append(f)
                continue
            if rawFeature['type'] == 'morphologicalFeature':
                f = feature.morphologicalFeature(rawFeature['subStr'], rawFeature['prefixOrSuffix'], rawFeature['tag'], rawFeature['name'])
                self.featureSet.append(f)
                continue
            if rawFeature['type'] == 'morphologicalBigramFeature':
                f = feature.morphologicalBigramFeature(rawFeature['subStr'], rawFeature['prefixOrSuffix'], rawFeature['tag'], rawFeature['prevTag'], rawFeature['name'])
                self.featureSet.append(f)
                continue
            if rawFeature['type'] == 'morphologicalTrigramFeature':
                f = feature.morphologicalTrigramFeature(rawFeature['subStr'], rawFeature['prefixOrSuffix'], rawFeature['tag'], rawFeature['prevTag'], rawFeature['prevPrevTag'], rawFeature['name'])
                self.featureSet.append(f)
                continue
            if rawFeature['type'] == 'tagUnigramFeature':
                f = feature.tagUnigramFeature(rawFeature['tag'], rawFeature['name'])
                self.featureSet.append(f)
                continue
            if rawFeature['type'] == 'tagBigramFeature':
                f = feature.tagBigramFeature(rawFeature['tag'], rawFeature['prevTag'], rawFeature['name'])
                self.featureSet.append(f)
                continue
            if rawFeature['type'] == 'tagTrigramFeature':
                f = feature.tagTrigramFeature(rawFeature['tag'], rawFeature['prevTag'], rawFeature['prevPrevTag'], rawFeature['name'])
                self.featureSet.append(f)
                continue
            raise 'unknown feature type'
        self.rawFeatures = None
                   
    def load(self, filename):
        """ load a model from file """ 
        with open(filename, 'rb') as inputFile:
            model = pickle.load(inputFile)
            self.dictionary = model.dictionary
            self.rawFeatures = model.rawFeatures
            self.featureNum = model.featureNum
            self.allSentences = model.allSentences
            self.sentenceNum = model.sentenceNum
            self.tagToFeatureIndices = model.tagToFeatureIndices
            self.tagToTagNgramFeatureIndices = model.tagToTagNgramFeatureIndices
            self.tagSet = model.tagSet
            self.lamda = model.lamda
            self.allWordsFeatureVecs = model.allWordsFeatureVecs
            self.v = model.v
            self.minFeatureCount = model.minFeatureCount
        
        self.loadFeaturesFromRaw()
    
    def readGoldenFile(self,wordfile, tagfile, numSentences):
        """ read training file """ 
        wf = open(wordfile,'rt')
        tf = open(tagfile,'rt')
        allSentences = []
        wlines = wf.readlines()
        tlines = tf.readlines()
        # build and save sentence objects
        for i in range(0,numSentences):
            # new sentence
            words = wlines[i].split()
            tags = tlines[i].split()
            s = sentence.sentence(words,tags)
            allSentences.append(s)
        wf.close()
        tf.close()
        # save data
        self.allSentences = allSentences
        self.sentenceNum = len(allSentences)
    
    def initModelParams(self,lamda,featureLevel):
        """ init dictionary (map words to (tag,count) tuple list
            i.e. dictionary[moshe] -> [(VB,1),(NN,2)] """
        allTagUnigrams = {}
        allTagBigrams = {}
        allTagTrigrams = {}
        for sentence in self.allSentences:
            for i in range(0,sentence.len):
                word = sentence.word(i)
                tag = sentence.tag(i)
                prevTag = sentence.tag(i - 1)
                prevPrevTag = sentence.tag(i - 2) 
                if (allTagUnigrams.has_key(tag)):
                    allTagUnigrams[tag] += 1
                else:
                    allTagUnigrams[tag] = 1
                if (allTagBigrams.has_key((tag,prevTag))):
                    allTagBigrams[(tag,prevTag)] += 1
                else:
                    allTagBigrams[(tag,prevTag)] = 1
                if (allTagTrigrams.has_key((tag,prevTag,prevPrevTag))):
                    allTagTrigrams[(tag,prevTag,prevPrevTag)] += 1
                else:
                    allTagTrigrams[(tag,prevTag,prevPrevTag)] = 1
                foundTag = False
                if self.dictionary.has_key(word):
                    # if this word already exists in the dictionary
                    for j in range(0,len(self.dictionary[word])):
                        (tag2,count) = self.dictionary[word][j]
                        # if this tag was already seen with this word
                        if tag2 == tag:
                            self.dictionary[word][j] = (tag, count + 1)
                            foundTag = True
                            break # break inner loop for searching a match for tag
                    if not foundTag:
                        # didn't find the tag - create a new entry for this word with this tag
                        self.dictionary[word].append((tag,1))
                else:
                    # didn't find word - create a new entry for it
                    self.dictionary[word] = [(tag,1)]
                if not foundTag:
                    # add tag to tagSet if needed
                    if self.tagSet.count(tag) == 0:
                        self.tagSet.append(tag)
        # TODO - cleanup tagset? => consider removing special tags such as -LRB- (which stands for the word "("
        self.lamda = lamda 
        # init feature set
        self.initFeatureSet(featureLevel,allTagUnigrams,allTagBigrams,allTagTrigrams)
        self.allWordsFeatureVecs = self.calcFeatureVecAllWords()
        
        # for debug
        self.featureNames = map(lambda x: x.name, self.featureSet)
        return
    
    def initModelFromFile(self, wordfile,tagfile,lamda,featureLevel,numSentences):
        """ init the model""" 
        # TODO - add offset (i.e. choose 5000 sentences starting from sentence 5001)
        self.readGoldenFile(wordfile,tagfile,numSentences)
        self.initModelParams(lamda,featureLevel)
    
    def addToFeatureMap(self,tag,index):
        """ add index to tag -> feature indices mapping"""
        if not self.tagToFeatureIndices.has_key(tag):
            self.tagToFeatureIndices[tag] = []
        self.tagToFeatureIndices[tag].append(index)
    
    def addToTagNgramFeatureMap(self, tag,index):
        """ add index to tag -> tag bigram indices 
            this is needed as for calculating P we need to calculate these features as well"""
        if not self.tagToTagNgramFeatureIndices.has_key(tag):
            self.tagToTagNgramFeatureIndices[tag] = []
        self.tagToTagNgramFeatureIndices[tag].append(index)
        
    def initFeatureSet(self,featureLevel,allTagUnigrams,allTagBigrams,allTagTrigrams):
        """calculate all features accordign to feature level"""
        # 1 => 001 => basic 
        # 2 => 010 => medium
        # 3 => 011 => medium + basic
        # 4 => 100 => advanced
        # 5 => 101 => advanced + basic
        # 6 => 110 => advanced + medium
        # 7 => 111 => advanced + medium + basic
        print "calculating features..."
        t1 = time.clock()
        if filter(lambda x: x == featureLevel, [1,3,5,7]):
            self.initBasicFeatures(allTagUnigrams,allTagBigrams,allTagTrigrams)
        if filter(lambda x: x == featureLevel, [2,3,6,7]):
            self.initMediumFeatures(allTagUnigrams,allTagBigrams,allTagTrigrams)
        if filter(lambda x: x == featureLevel, [4,5,6,7]):
            self.initAdvancedFeatures()
        self.featureNum = len(self.featureSet)
        t2 = time.clock()
        self.empiricalCounts = [f.count for f in self.featureSet];
        print "time to calc features:", t2 - t1, ", num of features is", self.featureNum
    
    def initBasicFeatures(self,allTagUnigrams,allTagBigrams,allTagTrigrams):
        for word in self.dictionary:
            for (tag,count) in self.dictionary[word]:
                if (count > self.minFeatureCount):
                    f = feature.unigramWordTagFeature(word,tag,count,"1_" + word + "_" + tag)
                    self.featureSet.append(f)
                    self.addToFeatureMap(tag, len(self.featureSet) - 1)
        for tag in allTagUnigrams:
            if allTagUnigrams[tag] > self.minFeatureCount:
                f = feature.tagUnigramFeature(tag,allTagUnigrams[tag],"2_" + tag)
                self.featureSet.append(f)
                self.addToFeatureMap(tag, len(self.featureSet) - 1)
                self.addToTagNgramFeatureMap(tag,len(self.featureSet) - 1)
        for (tag,prevTag) in allTagBigrams:
            if allTagBigrams[(tag,prevTag)] > self.minFeatureCount:
                f = feature.tagBigramFeature(tag,prevTag,allTagBigrams[(tag,prevTag)],"3_" + tag + "_" + prevTag)
                self.featureSet.append(f)
                self.addToFeatureMap(tag, len(self.featureSet) - 1)
                self.addToTagNgramFeatureMap(tag,len(self.featureSet) - 1)
        for (tag,prevTag,prevPrevTag) in allTagTrigrams:
            if allTagTrigrams[(tag,prevTag,prevPrevTag)] > self.minFeatureCount:
                f = feature.tagTrigramFeature(tag,prevTag,prevPrevTag,allTagTrigrams[(tag,prevTag,prevPrevTag)],"4_" + tag + "_" + prevTag + "_" + prevPrevTag)
                self.featureSet.append(f)
                self.addToFeatureMap(tag, len(self.featureSet) - 1)
                self.addToTagNgramFeatureMap(tag,len(self.featureSet) - 1)
        for tag in self.tagSet:
            if not self.tagToTagNgramFeatureIndices.has_key(tag):
                self.tagToTagNgramFeatureIndices[tag] = []
            if not self.tagToFeatureIndices.has_key(tag):
                self.tagToFeatureIndices[tag] = []
                
        
    def initMediumFeatures(self,allTagUnigrams,allTagBigrams,allTagTrigrams):
        suffixes = ['s','e','ed','y','n','ing','t','es','l','er','ly','ion','ted','ers','ent','ons','ies']
        prefixes = ['re','co','in','pr','de','st','con','di','pro']
        for s in suffixes:
            for tag in allTagUnigrams:
                if allTagUnigrams[tag] > self.minFeatureCount:
                    f = feature.morphologicalFeature(s,False,tag,"5_suff_" + s + "_" + tag)
                    self.featureSet.append(f)
                    self.addToFeatureMap(tag, len(self.featureSet) - 1)
            for (tag,prevTag) in allTagBigrams:
                if allTagBigrams[(tag,prevTag)] > self.minFeatureCount:
                    f = feature.morphologicalBigramFeature(s,False,tag,prevTag,"6_suff_" + s + "_" + tag + "_" + prevTag)
                    self.featureSet.append(f)
                    self.addToFeatureMap(tag, len(self.featureSet) - 1)
            for (tag,prevTag,prevPrevTag) in allTagTrigrams:
                if allTagTrigrams[(tag,prevTag,prevPrevTag)] > self.minFeatureCount:
                    f = feature.morphologicalTrigramFeature(s,False,tag,prevTag,prevPrevTag,"7_suff_" + s + "_" + tag + "_" + prevTag + "_" + prevPrevTag)
                    self.featureSet.append(f)
                    self.addToFeatureMap(tag, len(self.featureSet) - 1)
        for p in prefixes:
            for tag in allTagUnigrams:
                if allTagUnigrams[tag] > self.minFeatureCount:
                    f = feature.morphologicalFeature(p,True,tag,"8_pref_" + p + "_" + tag)
                    self.featureSet.append(f)
                    self.addToFeatureMap(tag, len(self.featureSet) - 1)
            for (tag,prevTag) in allTagBigrams:
                if allTagBigrams[(tag,prevTag)] > self.minFeatureCount:
                    f = feature.morphologicalBigramFeature(p,True,tag,prevTag,"9_pref_" + p + "_" + tag + "_" + prevTag)
                    self.featureSet.append(f)
                    self.addToFeatureMap(tag, len(self.featureSet) - 1)
            for (tag,prevTag,prevPrevTag) in allTagTrigrams:
                if allTagTrigrams[(tag,prevTag,prevPrevTag)] > self.minFeatureCount:
                    f = feature.morphologicalTrigramFeature(p,True,tag,prevTag,prevPrevTag,"10_pref_" + p + "_" + tag + "_" + prevTag + "_" + prevPrevTag)
                    self.featureSet.append(f)
                    self.addToFeatureMap(tag, len(self.featureSet) - 1)
        for tag in self.tagSet:
            if not self.tagToTagNgramFeatureIndices.has_key(tag):
                self.tagToTagNgramFeatureIndices[tag] = []
            if not self.tagToFeatureIndices.has_key(tag):
                self.tagToFeatureIndices[tag] = []
        self.calcEmpiricalCounts()
    
    def calcEmpiricalCounts(self):
        for sentence in self.sentenceNum:
            for i in range(0,sentence.len):
                print ""
            
    def initAdvancedFeatures(self):
        return
       
    def calcFeatureVecWord(self,word,tag,prevTag,prevPrevTag,subsetIndices = None):
        featureSet = self.featureSet
        if subsetIndices or (subsetIndices == []):
            featureSet = (self.featureSet[i] for i in subsetIndices)
        return map(lambda x: x.val(word,tag,prevTag,prevPrevTag),featureSet)
    
    def calcFeatureVecAllWords(self):
        """ claculate the feature vector for all words with thier golden tags
            here for saving calculations """
        allWordsFeatureVecs = []
        print "calculating feature vec for all words and golden tags..."
        s = 0
        t1 = time.clock()
        printStep = int(self.sentenceNum/5)
        for sentence in self.allSentences:
            for index in range(0,sentence.len):
                tag = sentence.tag(index)
                tagIndices = self.tagToFeatureIndices[tag]
                wordFeatureVec = numpy.zeros(self.featureNum)
                wordPartialVec = self.calcFeatureVecWord(sentence.word(index),tag,sentence.tag(index - 1),sentence.tag(index - 2),tagIndices)
                for i in tagIndices:
                    wordFeatureVec[i] = wordPartialVec[tagIndices.index(i)]
                allWordsFeatureVecs.append(wordFeatureVec)
            s = s + 1
            if self.verbose and ((s % printStep) == 1):
                print "\ts =",s,"out of",self.sentenceNum, "sentences, average iter time =",(time.clock() - t1)/s
        return allWordsFeatureVecs
                    
    def makeL(self):
        def L(*args):
            print "calculating L..."
            t1 = time.clock()
            v = args[0]
            print "v norm is ", math.sqrt(sum(v_i * v_i for v_i in v))
            val = 0
            i = 0
            s = 0
            printStep = int(self.sentenceNum/5)
            for sentence in self.allSentences:
                for index in range(0,sentence.len):
                    val = val + numpy.dot(v, self.allWordsFeatureVecs[i])
                    # calculate the inner sum of the second term
                    innerSum = 0
                    word = sentence.word(index);
                    prevTag = sentence.tag(index - 1)
                    prevPrevTag = sentence.tag(index - 2)
                    #for (tag,_) in self.dictionary[word]:
                    wordTags = [tag for (tag, _) in self.dictionary[word]]
                    for tag in self.tagSet:
                        exponent = 0
                        if tag in wordTags:
                            featureVec = self.calcFeatureVecWord(word,tag,prevTag,prevPrevTag,self.tagToFeatureIndices[tag])
                            sub_v = [v[j] for j in self.tagToFeatureIndices[tag]]
                            exponent = numpy.dot(featureVec, sub_v)
                        else:
                            # only calculate tag bigram and tag trigrams
                            featureVec = self.calcFeatureVecWord(word,tag,prevTag,prevPrevTag,self.tagToTagNgramFeatureIndices[tag])
                            sub_v = [v[j] for j in self.tagToTagNgramFeatureIndices[tag]]
                            exponent = numpy.dot(featureVec, sub_v)
                        try:
                            innerSum += math.exp(exponent)
                        except OverflowError:
                            print "overflow error: exponent =", exponent
                            raise
                    val = val - math.log(innerSum)
                    i = i + 1;
                s = s + 1;
                if self.verbose and ((s % printStep) == 1):
                    print "\ts =",s,"out of",self.sentenceNum, "sentences, average iter time =",(time.clock() - t1)/s
            sqNormV = sum(math.pow(v_k, 2) for v_k in v)
            val = val - ((self.lamda / 2) * sqNormV)
            t2 = time.clock()
            print "L val:",val
            print "time to calc L:", t2 - t1
            return val * (-1) # -1 as we want to maximize it, and fmin_bfgs only computes min
        return L;
    
    def makeGradientL(self):
        def gradientL(*args):
            print "calculating grad L..."
            t1 = time.clock()
            v = args[0]
            val = [empirical - (self.lamda * regularization) for (empirical,regularization) in \
                   zip(self.empiricalCounts,v)]
            printStep = int(self.sentenceNum/5)
            i = 0
            s = 0
            for sentence in self.allSentences:
                for index in range(0,sentence.len):
                    word = sentence.word(index)
                    prevTag = sentence.tag(index - 1)
                    prevPrevTag = sentence.tag(index - 2)
                    P = {}
                    allFeatureVecsByTags = {}
                    wordTags = [tag for (tag,_) in self.dictionary[word]]
                    for tag in self.tagSet:
                        if tag in wordTags:
                            featureVec = self.calcFeatureVecWord(word,tag,prevTag,prevPrevTag,self.tagToFeatureIndices[tag])
                            sub_v = [v[j] for j in self.tagToFeatureIndices[tag]]
                            power = numpy.dot(featureVec, sub_v)
                            allFeatureVecsByTags[tag] = featureVec
                            P[tag] = math.exp(power)
                        else:
                            featureVec = self.calcFeatureVecWord(word,tag,prevTag,prevPrevTag,self.tagToTagNgramFeatureIndices[tag])
                            sub_v = [v[j] for j in self.tagToTagNgramFeatureIndices[tag]]
                            power = numpy.dot(featureVec, sub_v)
                            allFeatureVecsByTags[tag] = featureVec
                            P[tag] = math.exp(power)
                    sumP = sum(P.values())
                    # expected count
                    for tag in self.tagSet:
                        if tag in wordTags:
                            for k in self.tagToFeatureIndices[tag]:
                                featureIndex = self.tagToFeatureIndices[tag].index(k)
                                f_k = allFeatureVecsByTags[tag][featureIndex]
                                val[k] -= (f_k * P[tag] / sumP)
                        else:
                            for k in self.tagToTagNgramFeatureIndices[tag]:
                                featureIndex = self.tagToTagNgramFeatureIndices[tag].index(k)
                                f_k = allFeatureVecsByTags[tag][featureIndex]
                                val[k] -= (f_k * P[tag] / sumP)
                    i = i + 1
                s = s + 1
                if self.verbose and ((s % printStep) == 1):
                    print "\ts =",s,"out of",self.sentenceNum, "sentences, average iter time =",(time.clock() - t1)/s
            newVal = map(lambda x: -1 * x, val) # -1 as we want to maximize it and fmin_bfgs only computes min
            t2 = time.clock()
            print "grad val:",newVal
            print "time to calc Grad L:", t2 - t1
            return scipy.array(newVal)
        return gradientL

    def trainModel(self):
        v = [0] * len(self.featureSet)
        try:
            # vopt = fmin_bfgs(self.makeL(), v, fprime=self.makeGradientL(), disp=self.verbose);
            vopt = fmin_l_bfgs_b(self.makeL(), v, fprime=self.makeGradientL(), disp=self.verbose)

        except OverflowError:
            print "math overflow error!"
            return 
        self.v = vopt[0]
    