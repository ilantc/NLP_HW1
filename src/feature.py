import re;
# base class - should implement derived classes
class feature:
    def __init__(self):
        return;
    def val(self,sentence,index,prevTag,prevPrevTag):
        return 0;

class morphologicalFeature(feature):
    
    def __init__(self,subStr,prefixOrSuffix,name):
        regex = subStr;
        if prefixOrSuffix:
            regex = "^" + regex;
        else:
            regex = regex + "$";
        self.f = lambda (sentence,index): re.search(regex, sentence.word[index])
        self.name = name;
    
    def val(self,sentence,index,prevTag,prevPrevTag):
        return self.f(sentence,index);