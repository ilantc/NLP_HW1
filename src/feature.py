import re;
# base class - should implement derived classes
class feature:
    def __init__(self):
        return;
    def val(self,word,tag,prevTag,prevPrevTag):
        return 0;
    def toRawObj(self):
        return

class morphologicalFeature(feature):
    
    def __init__(self,subStr,prefixOrSuffix,name):
        regex = subStr;
        if prefixOrSuffix:
            regex = "^" + regex;
        else:
            regex = regex + "$";
        self.f = lambda (word): re.search(regex, word)
        self.name = name;
        self.subStr = subStr;
        self.prefixOrSuffix = prefixOrSuffix
    
    def val(self,word,tag,prevTag,prevPrevTag):
        return 1 if self.f(word) else 0;
    
    def toRawObj(self):
        rawOBj = {'type':'morphologicalFeature', 'subStr': self.subStr, 'prefixOrSuffix' : self.prefixOrSuffix, 'name' : self.name};
        return rawOBj
    
    
class unigramWordTagFeature(feature):
    
    def __init__(self,word,tag,count,name):
        self.name = name;
        self.tag = tag;
        self.word = word;
        self.count = count
        self.f = lambda (_word,_tag): (_word == self.word) and (_tag == self.tag)  
        
    def val(self,word,tag,prevTag,prevPrevTag):
        return 1 if self.f((word,tag)) else 0;
    
    def toRawObj(self):
        rawOBj = {'type':'unigramWordTagFeature', 'name' : self.name, 'tag' : self.tag, 'word' : self.word}
        return rawOBj


class bigramWordTagFeature(feature):
    
    def __init__(self,word,tag,prevTag,count,name):
        self.name = name
        self.tag = tag
        self.prevTag = prevTag
        self.word = word
        self.count = count
        self.f = lambda (_word,_tag,_prevTag): (_word == self.word) and (_tag == self.tag) and (_prevTag == self.prevTag)
        
    def val(self,word,tag,prevTag,prevPrevTag):
        return 1 if self.f((word,tag,prevTag)) else 0;
    
    def toRawObj(self):
        rawOBj = {'type':'unigramWordTagFeature', 'name' : self.name, 'tag' : self.tag, 'word' : self.word, 'prevTag': self.prevTag}
        return rawOBj

class tagUnigramFeature(feature):
    
    def __init__(self,tag,count,name):
        self.name = name;
        self.tag = tag;
        self.count = count
        self.f = lambda (_tag): (_tag== self.tag)
        
    def val(self,word,tag,prevTag,prevPrevTag):
        return 1 if self.f((tag)) else 0;
    
    def toRawObj(self):
        rawOBj = {'type':'tagBigramFeature', 'name' : self.name, 'tag' : self.tag}
        return rawOBj


class tagBigramFeature(feature):
    
    def __init__(self,tag,prevTag,count,name):
        self.name = name;
        self.tag = tag;
        self.prevTag = prevTag;
        self.count = count
        self.f = lambda (_tag, _prevTag): (_tag== self.tag) and (_prevTag == self.prevTag)  
        
    def val(self,word,tag,prevTag,prevPrevTag):
        return 1 if self.f((tag,prevTag)) else 0;
    
    def toRawObj(self):
        rawOBj = {'type':'tagBigramFeature', 'name' : self.name, 'tag' : self.tag, 'prevTag' : self.prevTag}
        return rawOBj

class tagTrigramFeature(feature):
    
    def __init__(self,tag,prevTag,prevPrevTag,count,name):
        self.name = name;
        self.tag = tag;
        self.prevTag = prevTag;
        self.prevPrevTag = prevPrevTag
        self.count = count
        self.f = lambda (_tag, _prevTag,_prevPrevTag): (_tag== self.tag) and (_prevTag == self.prevTag) and (_prevPrevTag == self.prevPrevTag)
        
    def val(self,word,tag,prevTag,prevPrevTag):
        return 1 if self.f((tag,prevTag,prevPrevTag)) else 0;
    
    def toRawObj(self):
        rawOBj = {'type':'tagBigramFeature', 'name' : self.name, 'tag' : self.tag, 'prevTag' : self.prevTag, 'prevPrevTag' : self.prevPrevTag}
        return rawOBj

