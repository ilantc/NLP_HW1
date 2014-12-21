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

def processResults(allRes, allTags):
    # init dict 
    tag2counts = {}
    for tag in allTags:
        tag2counts[tag] = {'totalGold' : 0, 'totalPredicted' : 0, 'correctPred' : 0}
    for res in allRes:
        gold = res['gold']
        predicted = res['predicted']
        for i in range(0,len(gold)):
            goldTag = gold[i]
            predictedTag = predicted[i]
            tag2counts[goldTag]['totalGold'] += 1
            tag2counts[predictedTag]['totalPredicted'] += 1
            if goldTag == predictedTag:
                tag2counts[goldTag]['correctPred'] += 1
    
    totalFscoreCount = 0
    for tag in allTags:
        precision = tag2counts[tag]['correctPred'] / tag2counts[tag]['totalPredicted']
        recall = tag2counts[tag]['correctPred'] / tag2counts[tag]['totalGold'] 
        fScore = 2 * precision * recall / (precision + recall)
        totalFscoreCount += fScore
        tag2counts[tag] = {'fScore': fScore}
        print "tag", tag, "\tFscore", fScore
    totalFscoreCount = totalFscoreCount / len(allTags)
    
    print "average Fscore = ", totalFscoreCount
    
        

    

wordfile = "../data/sec2-21/sec2-21.words";
tagfile = "../data/sec2-21/sec2-21.pos";
lamda = 1;
featureLevel = 1; # basic
#featureLevel = 2; # med
#featureLevel = 4; # advanced
trainingOffset = 0;
trainingSentenceNum = 5000;
devSetOffset = trainingSentenceNum;
devSetSentenceNum = 2000;
testSetOffset = trainingSentenceNum + devSetSentenceNum;
testSetSentenceNum = 5000;

basicFeaturesMinWordCount = 10;
medFeaturesUniCount = 800
medFeaturesBiCount = 800
medFeaturesTriCount = 400
verbose = True

# model.initModelFromFile("../data/sec2-21/sec2-21.words", "../data/sec2-21/sec2-21.pos", lamda, featureLevel, numSentences)
# model.trainModel()
#model.save("advancedModel_5k_lambda_5.pkl")

for modelFileName in ['advancedModel_5k_lambda_0.5.pkl','advancedModel_5k_lambda_5.pkl','basicModel_5k_lambda_5.pkl','basicModel_5k_lambda_0.5.pkl']:
    modelFile = '../../NLP_HW1/models/' + modelFileName
    model = MEMMModel.MEMMModel(verbose,0,0,0,0)
    model.load(modelFile)
    print "model File Name:",modelFile
    model.summarize();
    t1 = time.clock()
    viterbi = ViterbiMEMMModel.ViterbiMEMMModel(model)
    viterbi.readGoldenFile(wordfile, tagfile, devSetSentenceNum, devSetOffset)
    allRes = viterbi.tagSentences()
    t2 = time.clock()
    print "time to infer: ", t2 - t1
    processResults(allRes)
    




#winsound.PlaySound("../yofi_sehel.wav",winsound.SND_FILENAME)