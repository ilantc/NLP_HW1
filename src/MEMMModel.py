import math
import operator
from scipy.optimize import fmin_bfgs

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
        self.allWordsFeatureVecs = self.calcFeatureVecAllWords()
        return;
    
    def initModelFromFile(self, file):
        self.readGoldenFile(file);
        self.initModelParams();
        
    def trainModel(self):
        v = [0] * len(self.featureSet);
        vopt = fmin_bfgs(self.makeL(), v, fprime=self.makeGradientL());
        
        
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
        def L(*args):
            val = 0;
            i = 0;
            for sentence in self.allSentences:
                for index in range(0,sentence.len):
                    val = val + self.product(args, self.allWordsFeatureVecs[i]);
                    # calculate the inner sum of the second term
                    innerSum = 0;
                    for tag in self.tagSet:
                        featureVec = self.calcFeatureVecWord(sentence,index,tag,sentence.tag(index - 1),sentence.tag(index - 2))
                        innerSum += math.exp(self.product(featureVec, args));
                    val = val - math.log(innerSum)
                    i = i + 1;
            sqNormV = sum(math.pow(v_k, 2) for v_k in args);
            val = val - ((self.lamda / 2) * sqNormV)
            return val * (-1); # -1 as we want to maximize it, and fmin_bfgs only computes min
        return L;
    
    def makeGradientL(self):
        def gradientL(*args):
            val = [0] * self.featureNum;
            for k in range(0,self.featureNum):
                i = 0;
                for sentence in self.allSentences:
                    for index in range(0,sentence.len):
                        # empirical counts
                        val[k] = val[k] + self.allWordsFeatureVecs[i][k];
                        
                        # expected count, first calc all P values
                        P = [0] * len(self.tagSet);
                        for y in range(0,len(self.tagSet)):
                            featureVec = self.calcFeatureVecWord(sentence,index,self.tagSet[y],sentence.tag(index - 1),sentence.tag(index - 2));
                            power = self.product(featureVec, args);
                            P[y] = math.exp(power);
                        sumP = sum(P);
                        expectedCount = 0;
                        for y in range(0,len(self.tagSet)):
                            f_k = self.featureSet[k].val(sentence,index,self.tagSet[y],sentence.tag(index - 1),sentence.tag(index - 2));
                            expectedCount = expectedCount + (f_k * P[y]/sumP);
                        val[k] = val[k] - expectedCount;
                        i = i + 1;
                val[k] = val[k] - (self.lamda*args[k]);
            newVal = val.map(lambda x: -1 * x, val); # -1 as we want to maximize it and fmin_bfgs only computes min
            return newVal;
        return gradientL;

    def product(self,vec1,vec2):
        return sum(map( operator.mul, vec1, vec2))