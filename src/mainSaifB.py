import MEMMModel;
import ViterbiMEMMModel;
import time;
import csv


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
            elif totalPred == 0: # and totalGold > 0
                precision = 1.0
                recall = 0.0
                fScore = 0.0
            else: # totalPred > 0
                precision =0.0
                recall = 1.0
                fScore = 0.0
        else:
            precision = float(correctPred) / float(totalPred)
            recall = float(correctPred) / float(totalGold)
            fScore = 2 * precision * recall / (precision + recall) 
        tag2counts[tag]['fScore'] =  fScore
        writer.writerow({'tag': tag, 'precision' : precision, 'recall': recall, 'fscore': fScore,'goldCount': totalGold,'predCount': totalPred,\
                        'correctPred': correctPred, 'fileName': filename, 'f * goldCount': fScore * totalGold })
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
testSetSentenceNum = 5000;
 
includeUniGram = True
includeBiGram = False
includeTriGram = False
 
basicFeaturesMinWordCount = 10;
medFeaturesUniCount = 800
medFeaturesBiCount = 800
medFeaturesTriCount = 400
verbose = True

csvfile = open('SaifBDev.csv', 'w')
fieldnames = ['tag', 'precision', 'recall', 'fscore','goldCount','predCount','correctPred','fileName','f * goldCount']
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()
path = '../../NLP_HW1/models/'

modelFileNameUni = path + 'basicModelUni.pkl'
modelFileNameBi  = path + 'basicModelUniBi.pkl'
modelFileNameTri = path + 'basicModel_5k_lambda_0.5.pkl'

modelUni = MEMMModel.MEMMModel(verbose,0,0,0,0)
modelUni.load(modelFileNameUni)
modelBi = MEMMModel.MEMMModel(verbose,0,0,0,0)
modelBi.load(modelFileNameBi)
modelTri = MEMMModel.MEMMModel(verbose,0,0,0,0)
modelTri.load(modelFileNameTri)

lamdas = []
# uni, bi, tri
lamdas.append([0.33,0.33,0.34])
lamdas.append([0.25,0.25,0.5])
lamdas.append([0.25,0.5,0.25])
lamdas.append([0.5,0.25,0.25])
lamdas.append([0.6,0.3,0.1])
lamdas.append([0.1,0.3,0.6])
lamdas.append([0.1,0.6,0.3])
lamdas.append([0.0,0.5,0.5])
lamdas.append([0.5,0.0,0.5])
lamdas.append([0.5,0.5,0.0])

for currLam in lamdas:
    t1 = time.clock()
    viterbi = ViterbiMEMMModel.ViterbiMEMMModel([modelUni,modelBi,modelTri],currLam)
    viterbi.readGoldenFile(wordfile, tagfile, devSetSentenceNum, devSetOffset)
    allRes = viterbi.tagSentences()
    t2 = time.clock()
    print "time to infer: ", t2 - t1
    processResults(allRes,modelUni.tagSet,'basic_' + str(currLam[2]) + "," + str(currLam[1]) + "," + str(currLam[0]),writer)

modelFileNameUni = path + 'advancedModelUni.pkl'
modelFileNameBi  = path + 'advancedModelUniBi.pkl'
modelFileNameTri = path + 'advancedModel_5k_lambda_0.5.pkl'

modelUni = MEMMModel.MEMMModel(verbose,0,0,0,0)
modelUni.load(modelFileNameUni)
modelBi = MEMMModel.MEMMModel(verbose,0,0,0,0)
modelBi.load(modelFileNameBi)
modelTri = MEMMModel.MEMMModel(verbose,0,0,0,0)
modelTri.load(modelFileNameTri)

for currLam in lamdas:
    t1 = time.clock()
    viterbi = ViterbiMEMMModel.ViterbiMEMMModel([modelUni,modelBi,modelTri],currLam)
    viterbi.readGoldenFile(wordfile, tagfile, devSetSentenceNum, devSetOffset)
    allRes = viterbi.tagSentences()
    t2 = time.clock()
    print "time to infer: ", t2 - t1
    processResults(allRes,modelUni.tagSet,'advanced_' + str(currLam[2]) + "," + str(currLam[1]) + "," + str(currLam[0]),writer)

csvfile.close()




#winsound.PlaySound("../yofi_sehel.wav",winsound.SND_FILENAME)