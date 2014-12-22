import math
import numpy
import sentence
import time 

class ViterbiMEMMModel:

    def __init__(self,MEMMModelList,lambdaList):
        self.MEMMModels= MEMMModelList
        self.lambdas = lambdaList
        # list of sentence objects (for training)
        self.tagSet = self.MEMMModels[0].tagSet

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
        v = d.values()
        k = d.keys()
        return k[v.index(maxv)]


    def tagSentences(self):

        #print "tagging", self.sentenceNum, "sentences... "
        s=0
        printStep = max(1,int(self.numberOfAllSentencesForTagging/5))
        allRes = []; 
        t1 = time.clock()
        for sentence in self.allSentencesForTagging[0:self.numberOfAllSentencesForTagging]:
            tags = self.tagSentence(sentence)
            allRes.append({'gold':sentence.tags[2:], 'predicted':tags});
            #print "time to infer sentence ",i,":", t2 - t1
            s=s+1
            if (s % printStep) == 1:
                print "\ts =",s,"out of",self.numberOfAllSentencesForTagging, "sentences, average iter time =",(time.clock() - t1)/s
        return allRes


    def readGoldenFile(self,wordfile, tagfile, numSentences, offset):
        """ read training file """
        wf = open(wordfile,'rt')
        tf = open(tagfile,'rt')
        allSentencesForTagging = []
        wlines = wf.readlines()
        tlines = tf.readlines()
        # build and save sentence objects
        for i in range(offset,offset + numSentences):
            # new sentence
            words = wlines[i].split()
            tags = tlines[i].split()
            s = sentence.sentence(words,tags)
            allSentencesForTagging.append(s)
        wf.close()
        tf.close()
        # save data
        self.allSentencesForTagging = allSentencesForTagging
        self.numberOfAllSentencesForTagging = len(allSentencesForTagging)

    def tagSentence (self, sentence):
        pi = []
        bp = []
        allTagSets = [['*'],['*']]
        for i in range(0,sentence.len):
            word = sentence.word(i)
            if (self.MEMMModels[0].dictionary.has_key(word)):
                currTagSet = [tag for (tag,_) in self.MEMMModels[0].dictionary[word]]
            else:
                currTagSet = self.MEMMModels[0].tagSet
            allTagSets.append(currTagSet)
            # calc q for all the possibilities
            q = {}
            for tagMinusTwo in allTagSets[i]:
                q[tagMinusTwo] = {}
                for tagMinusOne in allTagSets[i+1]:
                    q[tagMinusTwo][tagMinusOne] = {}

                    model2q = []
                    for model in self.MEMMModels:
                        norm = 0
                        model2q.append({})
                        for tag in allTagSets[i+2]:    
                            sub_v = [model.v[j] for j in model.tagToFeatureIndices[tag]]
                            featureVec = model.calcFeatureVecWord(word,tag,tagMinusOne,tagMinusTwo,model.tagToFeatureIndices[tag])
                            val = math.exp(numpy.dot(featureVec, sub_v))
                            model2q[-1][tag] = val
                            norm = norm + val
                        for tag in allTagSets[i+2]:
                            model2q[-1][tag] = model2q[-1][tag]/norm
                    for tag in allTagSets[i+2]:
                        val = 0
                        for model_i in range(0, len(self.MEMMModels)):
                            val += model2q[model_i][tag] * self.lambdas[model_i]
                        q[tagMinusTwo][tagMinusOne][tag] = val
            #print "time to calc q for word",i,time.clock() - q_t1
            pi.append({})
            bp.append({})
            for tag in allTagSets[i+2]:
                pi[i][tag]={}
                bp[i][tag]={}
                for tagMinusOne in allTagSets[i+1]:
                    innerPI = {}
                    for tagMinusTwo in allTagSets[i]:
                        if (i==0):
                            innerPI[tagMinusTwo] = q[tagMinusTwo][tagMinusOne][tag]
                        else:
                            innerPI[tagMinusTwo] = pi[i-1][tagMinusOne][tagMinusTwo]*q[tagMinusTwo][tagMinusOne][tag]
                    m=max(innerPI.values())
                    pi[i][tag][tagMinusOne] = m
                    bp[i][tag][tagMinusOne] = self.keywithmaxval(innerPI,m)
            #print "word #",i,":" , t2 - t1
        m=0
        t= [0] * sentence.len
        n=sentence.len-1
        for tagN in allTagSets[i+2]:
            for tagNMinusOne in allTagSets[i+1]:
                if pi[n][tagN][tagNMinusOne] > m :
                    m=pi[n][tagN][tagNMinusOne]
                    t[n] =  tagN
                    t[n-1] = tagNMinusOne
        for i in range(0,n-1):
            j=n-2-i
            t[j] = bp[j+2][t[j+2]][t[j+1]]

        numCorrect = 0
        for i in range(0,sentence.len):
            if (sentence.tag(i) == t[i]):
                numCorrect += 1
            #print sentence.word(i) + ' ' + sentence.tag(i) + ' result:' + t[i]
        # print "\tnum correct =",numCorrect,"/",sentence.len

        return t

