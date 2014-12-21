import math
import numpy
import sentence

class ViterbiMEMMModel:

    def __init__(self,MEMMModel):
        self.MEMMModel= MEMMModel
        # list of sentence objects (for training)
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
        v = d.values()
        k = d.keys()
        return k[v.index(maxv)]


    def tagSentences(self):

        #print "tagging", self.sentenceNum, "sentences... "
        i=1
        allRes = []; 
        for sentence in self.allSentencesForTagging[0:self.numberOfAllSentencesForTagging]:
            tags = self.tagSentence(sentence)
            allRes.append({'gold':sentence.tags[2:], 'predicted':tags});
            #print "time to infer sentence ",i,":", t2 - t1
            i=i+1
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
            if (self.MEMMModel.dictionary.has_key(word)):
                currTagSet = [tag for (tag,_) in self.MEMMModel.dictionary[word]]
            else:
                currTagSet = self.MEMMModel.tagSet
            allTagSets.append(currTagSet)
            
            # calc q for all the possibilities
            q = {}
            for tagMinusTwo in allTagSets[i]:
                q[tagMinusTwo] = {}
                for tagMinusOne in allTagSets[i+1]:
                    q[tagMinusTwo][tagMinusOne] = {}
                    norm = 0
                    for tag in allTagSets[i+2]:
                        sub_v = [self.optV[j] for j in self.MEMMModel.tagToFeatureIndices[tag]]
                        featureVec=self.MEMMModel.calcFeatureVecWord(word,tag,tagMinusOne,tagMinusTwo,self.MEMMModel.tagToFeatureIndices[tag])
                        q[tagMinusTwo][tagMinusOne][tag]=math.exp(numpy.dot(featureVec, sub_v))
                        norm = norm + q[tagMinusTwo][tagMinusOne][tag]
                    for tag in allTagSets[i+2]:
                        q[tagMinusTwo][tagMinusOne][tag]=q[tagMinusTwo][tagMinusOne][tag]/norm
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
                            innerPI[tagMinusTwo] = q[tagMinusTwo][tagMinusOne][tag]/norm
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

