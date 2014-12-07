class sentence:
    
    def __init__(self,words,tags):
        self.words = words;
        self.tags = ['*','*'] + tags;
        
    def tag(self,index):
        return self.tag(index + 2);
    
    def word(self,index):
        return self.word(index);