import MEMMModel;
import ViterbiMEMMModel;
import time;
import operator


def calcStat(wordDict,outfileSuff,outfilePref):
        
        suffToCount = {}
        prefToCount = {}
        for word in wordDict:
            # suff of len 1
            suff1 = word[-1]
            suff2 = word[-2:]
            suff3 = word[-3:]
            for suff in [suff1,suff2,suff3]:
                if suffToCount.has_key(suff):
                    suffToCount[suff] += 1
                else:
                    suffToCount[suff] = 1
            pref1 = word[0]
            pref2 = word[0:2]
            pref3 = word[0:3]
            for pref in [pref1,pref2,pref3]:
                if prefToCount.has_key(pref):
                    prefToCount[pref] += 1
                else:
                    prefToCount[pref] = 1
        sorted_suffToCount = sorted(suffToCount.items(), key=operator.itemgetter(1),reverse=True)
        sorted_prefToCount = sorted(prefToCount.items(), key=operator.itemgetter(1),reverse=True)
        f = open(outfileSuff,'w');
        f.write("\n".join('%s %s' % x for x in sorted_suffToCount))
        f.close()
        f = open(outfilePref,'w');
        f.write("\n".join('%s %s' % x for x in sorted_prefToCount))
        f.close()

# wordfile = "../data/sec2-21/small.words";
# tagfile = "../data/sec2-21/small.pos";
wordfile = "../data/sec2-21/sec2-21.words";
tagfile = "../data/sec2-21/sec2-21.pos";
lamda = 1;
featureLevel = 1; # basic
#featureLevel = 2; # med
#featureLevel = 4; # advanced
numSentences = 1000;
basicFeaturesMinWordCount = 0;
medFeaturesUniCount = 800
medFeaturesBiCount = 800
medFeaturesTriCount = 400
verbose = True
model = MEMMModel.MEMMModel(True,0,0,0,0)
# model.initModelFromFile("../data/sec2-21/sec2-21.words", "../data/sec2-21/sec2-21.pos", lamda, featureLevel, numSentences)
# model.trainModel()
model.load('../../NLP_HW1/models/advancedModel_5k_lambda_0.5.pkl')
t1 = time.clock()
viterbi = ViterbiMEMMModel.ViterbiMEMMModel(model,numSentences)
viterbi.readGoldenFile(wordfile, tagfile, 1000, 5000)
allRes=viterbi.tagSentences()
averagePrecision = 0.0;

for res in allRes:
    gold = res['gold']
    predicted = res['predicted']
    sumEq = sum([x == y for (x,y) in zip(gold,predicted)])
    #print "sumEq =", sumEq, "sum/len =",float(sumEq)/float(len(gold)) 
    averagePrecision += float(sumEq)/float(len(gold)) 
averagePrecision = averagePrecision/float(len(allRes))
print ("average precision =",averagePrecision)
    
# print tags
t2 = time.clock()
print "time to infer: ", t2 - t1
#model.save("advancedModel_5k_lambda_5.pkl")
model.summarize();


#winsound.PlaySound("../yofi_sehel.wav",winsound.SND_FILENAME)