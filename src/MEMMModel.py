import math
import operator
from itertools import product

class MEMMModel:
    
    startSentenceTag = '*';
    
    def __init__(self):
        # dirctionary is amapping from word to (tag,count) tuple list
        self.dictionary = {};
        # an ordered list of functions from (sentence,index,tag_i-1.tag_i-2) to bool
        self.featureSet = [];
        self.featureNum = 0;
        
        self.allSentences = [];
        self.sentenceNum = 0;
        # all possible tags
        self.tagSet = [MEMMModel.startSentenceTag];
        self.lamda = 0;
        self.v = [];
        
    def reset(self):
        self.dictionary = {};
        self.featureSet = [];
        self.featureNum = 0;
        self.allSentences = [];
        self.sentenceNum = 0;
        self.tagSet = [MEMMModel.startSentenceTag];
        self.lamda = 0;
        self.v = [];
        
    def readGoldenFile(self,file):
        # TODO implement
        allSentences = [];
        self.allSentences = allSentences;
    
    def initModelParams(self):
        # init word difctionary
        # init feature set
        return;
    
    def initModelFromFile(self, file):
        self.readGoldenFile(file);
        self.initModelParams();
        
    def trainModel(self):
        v = [0] * len(self.featureSet)
        notConverged = True;
        gradient = [0] * self.featureNum;
        while notConverged:
            for k in range(0, self.featureNum): 
                empiricalCount = 0;
                expectedCount = 0;
                for sentence in self.allSentences:
                    for index in range(0,sentence.len):
                        # consider keeping a cache of calculated features
                        empiricalCount += self.featureSet[k].val(sentence,index,sentence.tag(index),sentence.tag(index - 1),sentence.tag(index - 2));
                        pDenominator = 0;
                        for tag in self.tagSet:
                            pDenominator = pDenominator + math.exp(product())  
                            
                gradient[k] = empiricalCount - expectedCount;
    
    def calcFeatureVecWord(self,sentence,index,tag,prevTag,prevPrevTag):
        return map(lambda x: x.val(sentence,index,tag,prevTag,prevPrevTag),self.featureSet);
    
    def calcFeatureVecAllWords(self):
        allWordsFeatureVecs = [];
        for sentence in self.allSentences:
            for index in range(0,sentence.len):
                wordFeatureVec = self.calcFeatureVecWord(sentence,index,sentence.tag(index),sentence.tag(index - 1),sentence.tag(index - 2));
                allWordsFeatureVecs.append(wordFeatureVec);
        return allWordsFeatureVecs;
                    
    def makeL(self):
        allWordsFeatureVecs = self.calcFeatureVecAllWords()
        def L(*args):
            val = 0;
            i = 0;
            for sentence in self.allSentences:
                for index in range(0,sentence.len):
                    val = val + self.product(args, allWordsFeatureVecs[i]);
                    # calculate the inner sum of the second term
                    innerSum = 0;
                    for tag in self.tagSet:
                        featureVec = self.calcFeatureVecWord(sentence,index,tag,sentence.tag(index - 1),sentence.tag(index - 2))
                        innerSum += math.exp(self.product(featureVec, args));
                    val = val - math.log(innerSum)
                    i = i + 1;
            sqNormV = sum(math.pow(v_k, 2) for v_k in args);
            val = val - ((self.lamda / 2) * sqNormV)
            return val;
        return L;
    
    def makeGradientL(self):
        

          
    def product(self,vec1,vec2):
        return sum(map( operator.mul, vec1, vec2))