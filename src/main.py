import MEMMModel;
import time;


# wordfile = "../data/sec2-21/small.words";
# tagfile = "../data/sec2-21/small.pos";
wordfile = "../data/sec2-21/sec2-21.words";
tagfile = "../data/sec2-21/sec2-21.pos";
lamda = 10;
featureLevel = 1; # basic
numSentences = 400;
verbose = True

print "initializing..."
t1 = time.clock()
model = MEMMModel.MEMMModel(verbose);
model.initModelFromFile(wordfile,tagfile,lamda,featureLevel,numSentences)
t2 = time.clock();
print "time to initialize: ", t2 - t1
model.summarize();
# model.show();
model.trainModel();
print model.v