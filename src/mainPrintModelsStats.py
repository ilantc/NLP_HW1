import MEMMModel;



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
includeBiGram = True
includeTriGram = True
 
basicFeaturesMinWordCount = 10;
medFeaturesUniCount = 800
medFeaturesBiCount = 800
medFeaturesTriCount = 400
verbose = True
 
model = MEMMModel.MEMMModel(verbose,basicFeaturesMinWordCount,medFeaturesUniCount,medFeaturesBiCount,medFeaturesTriCount)
model.initModelFromFile("../data/sec2-21/sec2-21.words", "../data/sec2-21/sec2-21.pos", lamda, featureLevel, \
                        trainingSentenceNum,includeUniGram,includeBiGram,includeTriGram)
model.summarize()

print "=================================================="
featureLevel = 2
model = MEMMModel.MEMMModel(verbose,basicFeaturesMinWordCount,medFeaturesUniCount,medFeaturesBiCount,medFeaturesTriCount)
model.initModelFromFile("../data/sec2-21/sec2-21.words", "../data/sec2-21/sec2-21.pos", lamda, featureLevel, \
                        trainingSentenceNum,includeUniGram,includeBiGram,includeTriGram)
model.summarize()


