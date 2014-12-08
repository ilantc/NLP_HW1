import re;
# base class - should implement derived classes
class feature:
    def __init__(self):
        return;
    def val(self,sentence,index,tag,prevTag,prevPrevTag):
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
    
    def val(self,sentence,index,tag,prevTag,prevPrevTag):
        return 1 if self.f(sentence,index) else 0;
    
    
class unigramWordTagFeature(feature):
    
    def __init__(self,word,tag,name):
        self.f = lambda (sentence,index,tag): (sentence.word(index) == word) and (sentence.tag(index) == tag)  
        self.name = name;
    
    def val(self,sentence,index,tag,prevTag,prevPrevTag):
        return 1 if self.f((sentence,index,tag)) else 0;