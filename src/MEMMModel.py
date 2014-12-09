import math
import operator
import sentence
from scipy.optimize import fmin_bfgs
import scipy
import feature
import time

class MEMMModel:
    
    startSentenceTag = '*';
    
    def __init__(self,verbose):
        # dirctionary is amapping from word to (tag,count) tuple list
        self.dictionary = {};
        # an ordered list of functions from (sentence,index,tag_i-1.tag_i-2) to bool
        self.featureSet = [];
        self.featureNum = 0;
        
        self.allSentences = [];
        self.sentenceNum = 0;
        # all possible tags
        self.tagSet = [];
        self.lamda = 0;
        self.allWordsFeatureVecs = [];
        self.verbose = verbose;
    
    def summarize(self):
        print "MODEL SUMMARY:"
        print   "\tnum sentences =", self.sentenceNum, \
              "\n\tnum features  =", self.featureNum, \
              "\n\tnum tags      =", len(self.tagSet), \
              "\n\tlamda         =", self.lamda, \
              "\n\tnum words     =", len(self.allWordsFeatureVecs) 
    
    def show(self):
        print "sentences are:"
        for sentence in self.allSentences:
            print sentence.toString()
        print self.dictionary
        for f in self.featureSet:
            print f.name
    
    def reset(self):
        self.dictionary = {};
        self.featureSet = [];
        self.featureNum = 0;
        self.allSentences = [];
        self.sentenceNum = 0;
        self.tagSet = [];
        self.lamda = 0;
        self.allWordsFeatureVecs = [];
        self.v = 0;
        
    def readGoldenFile(self,wordfile, tagfile, numSentences):
        wf = open(wordfile,'rt');
        tf = open(tagfile,'rt');
        allSentences = [];
        wlines = wf.readlines()
        tlines = tf.readlines()
        for i in range(0,numSentences):
            # new sentence
            words = wlines[i].split();
            tags = tlines[i].split();
            s = sentence.sentence(words,tags);
            allSentences.append(s);
        wf.close();
        tf.close();
        self.allSentences = allSentences;
        self.sentenceNum = len(allSentences);
    
    def initModelParams(self,lamda,featureLevel):
        # init dictionary (map words to (tag,count) tuple list
        # i.e. dictionary[moshe] -> [(VB,1),(NN,2)]
        for sentence in self.allSentences:
            for i in range(0,sentence.len):
                word = sentence.word(i);
                tag = sentence.tag(i);
                foundTag = False;
                if self.dictionary.has_key(word):
                    for j in range(0,len(self.dictionary[word])):
                        (tag2,count) = self.dictionary[word][j] 
                        if tag2 == tag:
                            self.dictionary[word][j] = (tag, count + 1)
                            foundTag = True
                            break
                    if not foundTag:
                        self.dictionary[word].append((tag,1))
                else:
                    self.dictionary[word] = [(tag,1)];
                if not foundTag:
                    if self.tagSet.count(tag) == 0:
                        self.tagSet.append(tag)
        # TODO - cleanup tagset?
        self.lamda = lamda 
        # init feature set
        self.initFeatureSet(featureLevel)
        self.allWordsFeatureVecs = self.calcFeatureVecAllWords()
        return;
    
    def initModelFromFile(self, wordfile,tagfile,lamda,featureLevel,numSentences):
        # TODO - add offset
        self.readGoldenFile(wordfile,tagfile,numSentences);
        self.initModelParams(lamda,featureLevel);
  
    def initFeatureSet(self,featureLevel):
        # 1 => 001 => basic 
        # 2 => 010 => medium
        # 3 => 011 => medium + basic
        # 4 => 100 => advanced
        # 5 => 101 => advanced + basic
        # 6 => 110 => advanced + medium
        # 7 => 111 => advanced + medium + basic
        print "calculating features..."
        t1 = time.clock();
        if filter(lambda x: x == featureLevel, [1,3,5,7]):
            self.initBasicFeatures();
        if filter(lambda x: x == featureLevel, [2,3,6,7]):
            self.initMediumFeatures();
        if filter(lambda x: x == featureLevel, [4,5,6,7]):
            self.initAdvancedFeatures();
        t2 = time.clock();
        print "time to calc features:", t2 - t1;
    
    def initBasicFeatures(self):
        for word in self.dictionary:
            wordCount = sum(count for (tag,count) in self.dictionary[word]);
            if (wordCount > 10):
                for (tag,count) in self.dictionary[word]:
                    f = f = feature.unigramWordTagFeature(word,tag,word + "_" + tag);
                    self.featureSet.append(f)
                    self.featureNum = self.featureNum + 1;
#         f = feature.unigramWordTagFeature("the","DT","the_dt");
#         self.featureSet.append(f)
#         f = feature.unigramWordTagFeature("the","NN","the_nn");
#         self.featureSet.append(f)
#         f = feature.unigramWordTagFeature("plays","VBZ","plays_vbz");
#         self.featureSet.append(f)
#         self.featureNum = self.featureNum + 3;
        
        
    def initMediumFeatures(self):
        return
    
    def initAdvancedFeatures(self):
        return
       
    def calcFeatureVecWord(self,sentence,index,tag,prevTag,prevPrevTag):
        return map(lambda x: x.val(sentence.word(index),tag,prevTag,prevPrevTag),self.featureSet);
    
    def calcFeatureVecAllWords(self):
        allWordsFeatureVecs = [];
        for sentence in self.allSentences:
            for index in range(0,sentence.len):
                wordFeatureVec = self.calcFeatureVecWord(sentence,index,sentence.tag(index),sentence.tag(index - 1),sentence.tag(index - 2));
                allWordsFeatureVecs.append(wordFeatureVec);
        return allWordsFeatureVecs;
                    
    def makeL(self):
        def L(*args):
            print "calculating L..."
            t1 = time.clock();
            v = args[0];
            print "v norm is ", math.sqrt(sum(v_i * v_i for v_i in v));
            val = 0;
            i = 0;
            s = 0;
            for sentence in self.allSentences:
                for index in range(0,sentence.len):
                    val = val + self.product(v, self.allWordsFeatureVecs[i]);
                    # calculate the inner sum of the second term
                    innerSum = 0;
                    for tag in self.tagSet:
                        featureVec = self.calcFeatureVecWord(sentence,index,tag,sentence.tag(index - 1),sentence.tag(index - 2))
                        exponent = self.product(featureVec, v);
                        # print "exponent is: ", exponent
                        try:
                            innerSum += math.exp(exponent);
                        except OverflowError:
                            print "overflow error: exponent =", exponent
                            raise
                    val = val - math.log(innerSum)
                    i = i + 1;
                s = s + 1;
                if self.verbose and ((s % 20) == 1):
                    print "\ts =",s,"out of",self.sentenceNum, "sentences, average iter time =",(time.clock() - t1)/s;
            sqNormV = sum(math.pow(v_k, 2) for v_k in v);
            val = val - ((self.lamda / 2) * sqNormV)
            t2 = time.clock();
            print "time to calc L:", t2 - t1;
            return val * (-1); # -1 as we want to maximize it, and fmin_bfgs only computes min
        return L;
    
    def makeGradientL(self):
        def gradientL(*args):
            print "calculating grad L..."
            t1 = time.clock();
            v = args[0]
            val = [0] * self.featureNum;
            
            i = 0;
            s = 0;
            for sentence in self.allSentences:
                for index in range(0,sentence.len):
                    P = [0] * len(self.tagSet);
                    for y in range(0,len(self.tagSet)):
                        featureVec = self.calcFeatureVecWord(sentence,index,self.tagSet[y],sentence.tag(index - 1),sentence.tag(index - 2));
                        power = self.product(featureVec, v);
                        P[y] = math.exp(power);
                    sumP = sum(P);
                    for k in range(0,self.featureNum):
                        # empirical counts
                        val[k] = val[k] + self.allWordsFeatureVecs[i][k];
                        # expected count
                        expectedCount = 0;
                        for y in range(0,len(self.tagSet)):
                            f_k = self.featureSet[k].val(sentence.word(index),self.tagSet[y],sentence.tag(index - 1),sentence.tag(index - 2));
                            expectedCount = expectedCount + (f_k * P[y]/sumP);
                        val[k] = val[k] - expectedCount;
                        val[k] = val[k] - (self.lamda*v[k]);
                    i = i + 1;
                s = s + 1;
                if self.verbose and ((s % 20) == 1):
                    print "\ts =",s,"out of",self.sentenceNum, "sentences, average iter time =",(time.clock() - t1)/s;
            newVal = map(lambda x: -1 * x, val); # -1 as we want to maximize it and fmin_bfgs only computes min
            t2 = time.clock();
            print "time to calc Grad L:", t2 - t1;
            return scipy.array(newVal);
        return gradientL;

    def product(self,vec1,vec2):
        return sum(map( operator.mul, vec1, vec2))
    
    def trainModel(self):
        v = [0.1] * len(self.featureSet);
        try:
            vopt = fmin_bfgs(self.makeL(), v, fprime=self.makeGradientL(), disp=self.verbose);
        except OverflowError:
            print "math overflow error!"
            return 
        self.v = vopt;
    