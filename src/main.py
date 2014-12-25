import MEMMModel;
import ViterbiMEMMModel;
import time;
import operator
import math
import csv


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

def processResults(allRes, allTags,filename,writer):
    # init dict 
    tag2counts = {}
    confMatrix = {}
    for tag in allTags:
        tag2counts[tag] = {'totalGold' : 0, 'totalPredicted' : 0, 'correctPred' : 0}
        confMatrix[tag] = {};
        for tag2 in allTags:
            confMatrix[tag][tag2] = 0;
        
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
            confMatrix[predictedTag][goldTag] += 1
    
    for tag in allTags:
        correctPred = tag2counts[tag]['correctPred']
        totalPred = tag2counts[tag]['totalPredicted']
        totalGold = tag2counts[tag]['totalGold']
        print "tag", tag, "correctPred", correctPred, "totalPred", totalPred, "totalGold", totalGold
        if correctPred == 0:
            if (totalPred == 0) and (totalGold == 0):
                precision = 1.0
                recall = 1.0
                fScore = 1.0
            elif totalPred == 0:
                precision = 1.0
                recall = 0.0
                fScore = 0.0
            else:
                precision =0.0
                recall = 1.0
                fScore = 0.0
        else:
            precision = float(correctPred) / float(totalPred)
            recall = float(correctPred) / float(totalGold)
            fScore = 2 * precision * recall / (precision + recall) 
        tag2counts[tag]['fScore'] =  fScore
        writer.writerow({'tag': tag, 'precision' : precision, 'recall': recall, 'fscore': fScore,'goldCount': totalGold,'predCount': totalPred,\
                        'correctPred': correctPred, 'fileName': filename })
    #print confusion matrix
    confMatFileName = filename + '_confMatrix.csv'
    csvfile2 = open(confMatFileName, 'w')
    fieldnames = ["predictedTag"] + allTags
    writer2 = csv.DictWriter(csvfile2, fieldnames=fieldnames)
    writer2.writeheader()
    for tag in confMatrix:
        row = confMatrix[tag];
        row['predictedTag'] = tag
        writer2.writerow(row)
    csvfile2.close

    

wordfile = "../data/sec2-21/sec2-21.words";
tagfile = "../data/sec2-21/sec2-21.pos";
 
lamda = 0.5;
featureLevel = 1; # basic
#featureLevel = 2; # med
#featureLevel = 4; # advanced
trainingOffset = 0;
trainingSentenceNum = 5000;
devSetOffset = trainingSentenceNum;
devSetSentenceNum = 1500;
testSetOffset = trainingSentenceNum + devSetSentenceNum;
testSetSentenceNum = 2000;
 
includeUniGram = True
includeBiGram = False
includeTriGram = False
 
basicFeaturesMinWordCount = 10;
medFeaturesUniCount = 800
medFeaturesBiCount = 800
medFeaturesTriCount = 400
verbose = True
# 
# model = MEMMModel.MEMMModel(verbose,basicFeaturesMinWordCount,medFeaturesUniCount,medFeaturesBiCount,medFeaturesTriCount)
# model.initModelFromFile("../data/sec2-21/sec2-21.words", "../data/sec2-21/sec2-21.pos", lamda, featureLevel, \
#                         trainingSentenceNum,includeUniGram,includeBiGram,includeTriGram)
# model.trainModel()
# model.save("basicModelUni.pkl")


csvfile = open('tmp.csv', 'w')
fieldnames = ['tag', 'precision', 'recall', 'fscore','goldCount','predCount','correctPred','fileName']
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()
allFiles = ['advancedModel_5k_lambda_0.5.pkl','advancedModel_5k_lambda_5.pkl','basicModel_5k_lambda_0.5.pkl','basicModel_5k_lambda_5.pkl']
allFiles = ['basicModel_5k_lambda_5.pkl']
devSetSentenceNum = 20;
for modelFileName in allFiles:
    modelFile = '../../NLP_HW1/models/' + modelFileName
    model = MEMMModel.MEMMModel(verbose,0,0,0,0)
    model.load(modelFile)
    print "model File Name:",modelFile
    model.summarize();
    t1 = time.clock()
    viterbi = ViterbiMEMMModel.ViterbiMEMMModel([model],[1])
    viterbi.readGoldenFile(wordfile, tagfile, devSetSentenceNum, devSetOffset)
    allRes = viterbi.tagSentences()
    t2 = time.clock()
    print "time to infer: ", t2 - t1
    processResults(allRes,model.tagSet,modelFileName,writer)
csvfile.close()




#winsound.PlaySound("../yofi_sehel.wav",winsound.SND_FILENAME)