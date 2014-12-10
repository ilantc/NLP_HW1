import MEMMModel;
import time;
import winsound;
import cProfile

# wordfile = "../data/sec2-21/small.words";
# tagfile = "../data/sec2-21/small.pos";
wordfile = "../data/sec2-21/sec2-21.words";
tagfile = "../data/sec2-21/sec2-21.pos";
lamda = 0.5;
featureLevel = 1; # basic
numSentences = 5;
basicFeaturesMinWordCount = 2;
verbose = True

print "initializing..."
t1 = time.clock()
model = MEMMModel.MEMMModel(verbose,basicFeaturesMinWordCount);
model.initModelFromFile(wordfile,tagfile,lamda,featureLevel,numSentences)
t2 = time.clock();
print "time to initialize: ", t2 - t1
model.summarize();
# model.show();
t1 = time.clock()
#cProfile.run('model.trainModel()')
model.trainModel()
t2 = time.clock()
print "time to train: ", t2 - t1
print model.v
model.summarize();
model.save('tryModel.pkl')
model2 = MEMMModel.MEMMModel(verbose,basicFeaturesMinWordCount);
model2.load('tryModel.pkl')
print model2.v
model2.summarize()

winsound.PlaySound("../yofi_sehel.wav",winsound.SND_FILENAME)