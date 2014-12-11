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
numSentences = 5000;
basicFeaturesMinWordCount = 15;
verbose = True

print "initializing..."
t1 = time.clock()
model = MEMMModel.MEMMModel(verbose,basicFeaturesMinWordCount);
model.initModelFromFile(wordfile,tagfile,lamda,featureLevel,numSentences)
t2 = time.clock();
print "time to initialize: ", t2 - t1
model.summarize();
t1 = time.clock()
#cProfile.run('model.trainModel()')
model.trainModel()
t2 = time.clock()
print "time to train: ", t2 - t1
model.save("basicModel_5k.pkl")
model.summarize();


winsound.PlaySound("../yofi_sehel.wav",winsound.SND_FILENAME)