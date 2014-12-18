import math
from scipy.optimize import fmin_l_bfgs_b
import scipy
import numpy
import feature
import sentence
import time
import cPickle as pickle

class ViterbiMEMMModel:

    def __init__(self,MEMMModel,sentenceNum):
        self.MEMMModel= MEMMModel
        # list of sentence objects (for training)
        self.allSentences = MEMMModel.allSentences
        self.sentenceNum = sentenceNum
        self.tagSet = MEMMModel.tagSet
        self.optV = MEMMModel.v

    def readFile(self,wordfile, tagfile, numSentences):
        """ read validation file """
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
        self.allSentences = allSentences
        self.sentenceNum = len(allSentences)



    def keywithmaxval(self,d,maxv):
         v=list(d.values())
         k=list(d.keys())
         return k[v.index(maxv)]


    def tagSentences(self):
        tags = []
        print "tagging", self.sentenceNum, "sentences... "
        i=1
        for sentence in self.allSentences[0:self.sentenceNum]:
            t1 = time.clock()
            tags = tags + self.tagSentence(sentence)
            t2 = time.clock()
            print "time to infer sentence ",i,":", t2 - t1
            i=i+1
        return tags

    def tagSentence (self, sentence):
        pi = []
        bp = []
        for i in range(0,sentence.len):
            t1 = time.clock()

            tagSetMinusOne = []
            tagSetMinusTwo = []
            if (i==0):
                tagSetMinusOne.append('*')
                tagSetMinusTwo.append('*')
            if (i==1):
                tagSetMinusOne=self.tagSet
                tagSetMinusTwo.append('*')
            if (i>1):
                tagSetMinusOne=self.tagSet
                tagSetMinusTwo=self.tagSet

            word = sentence.word(i)
            # calc q for all the possibilities
            q = {}
            for tagMinusTwo in tagSetMinusTwo:
                q[tagMinusTwo] = {}
                for tagMinusOne in tagSetMinusOne:
                    q[tagMinusTwo][tagMinusOne] = {}
                    norm = 0
                    for tag in self.tagSet:
                        featureVec=self.MEMMModel.calcFeatureVecWord(word,tag,tagMinusOne,tagMinusTwo)
                        q[tagMinusTwo][tagMinusOne][tag]=math.exp(numpy.dot(featureVec, self.optV))
                        norm = norm + q[tagMinusTwo][tagMinusOne][tag]
                    for tag in self.tagSet:
                        q[tagMinusTwo][tagMinusOne][tag]=q[tagMinusTwo][tagMinusOne][tag]/norm
            pi.append({})
            bp.append({})
            for tag in self.tagSet:
                pi[i][tag]={}
                bp[i][tag]={}
                for tagMinusOne in tagSetMinusOne:
                    innerPI = {}
                    for tagMinusTwo in tagSetMinusTwo:
                        if (i==0):
                            innerPI[tagMinusTwo] = q[tagMinusTwo][tagMinusOne][tag]/norm
                        else:
                            innerPI[tagMinusTwo] = pi[i-1][tagMinusOne][tagMinusTwo]*q[tagMinusTwo][tagMinusOne][tag]
                    m=max(innerPI.values())
                    pi[i][tag][tagMinusOne] = m
                    bp[i][tag][tagMinusOne] = self.keywithmaxval(innerPI,m)
            t2 = time.clock()
            print "word #",i,":" , t2 - t1
        m=0
        t=[0 for x in range(sentence.len)]
        n=sentence.len-1
        for tagN in self.tagSet:
            for tagNMinusOne in self.tagSet:
                if pi[n][tagN][tagNMinusOne] > m :
                    m=pi[n][tagN][tagNMinusOne]
                    t[n] =  tagN
                    t[n-1] = tagNMinusOne
        for i in range(0,n-1):
            j=n-2-i
            t[j] = bp[j+2][t[j+2]][t[j+1]]

        for i in range(0,sentence.len):
            print sentence.word(i) + ' ' + sentence.tag(i) + 'result:' + t[i]


        return t

