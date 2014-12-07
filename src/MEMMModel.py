
class MEMMModel:
    
    startSentenceTag = '*';
    
    def __init__(self):
        # dirctionary is amapping from word to (tag,count) tuple list
        self.dictionary = {};
        # an ordered list of functions from (sentence,index,tag_i-1.tag_i-2) to bool
        self.featureSet = [];
        # all possible tags
        self.tagSet = [MEMMModel.startSentenceTag];
        
    def reset(self):
        self.dictionary = {};
        self.featureSet = [];
        self.tagSet = [MEMMModel.startSentenceTag]
        
    def readGoldenFile(self,file):
        # TODO implement
        allSentences = [];
        self.allSentences = allSentences;
    
    def initModelParams(self):
        # init word difctionary
        # init feature set
        return;
    
    def initModelFromFile(self, file):
        self.readGoldenFile(file);
        self.initModelParams();
        
    def trainModel(self):
        v = [0] * len(self.featureSet)
        notConverged = True;
        gradient = [0] * len(self.featureSet);
        while notConverged:
            for k in range(0, len(self.featureSet)): 
                empiricalCount = 0;
                for sentence in self.allSentences:
                    for index in range(0,len(sentence)):
                        empiricalCount += self.featureSet[k].val(sentence,index,sentence.tag(index - 1),sentence.tag(index - 2));
                