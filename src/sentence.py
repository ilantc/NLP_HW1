class sentence:
    
    def __init__(self,words,tags):
        self.words = words;
        self.tags = ['*','*'] + tags;
        self.len = len(words);
        
    def tag(self,index):
        return self.tags[index + 2];
    
    def word(self,index):
        return self.words[index];
    
    def toString(self):
        mystr = ""
        mystr += ' '.join(word for word in self.words) + "\n"
        tagsToShow = self.tags[2:];
        mystr += ' '.join(tag for tag in tagsToShow) + "\n"
        return mystr;
        
