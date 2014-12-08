import MEMMModel;


wordfile = "../data/sec2-21/small.words";
tagfile = "../data/sec2-21/small.pos";
lamda = 0.5;
featureLevel = 1; # basic

model = MEMMModel.MEMMModel();
model.initModelFromFile(wordfile,tagfile,lamda,featureLevel)
# model.show();
model.trainModel();
print model.v