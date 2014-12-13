import MEMMModel;
import time;
import winsound;
import cProfile
import operator


def calcStat(wordDict,outfileSuff,outfilePref):
        
        suffToCount = {}
        prefToCount = {}
        for word in wordDict:
            # suff of len 1
            suff1 = word[-1]
            suff2 = word[-2:]
            suff3 = word[-3:]
            for suff in [suff1,suff2,suff3]:
                if suffToCount.has_key(suff):
                    suffToCount[suff] += 1
                else:
                    suffToCount[suff] = 1
            pref1 = word[0]
            pref2 = word[0:2]
            pref3 = word[0:3]
            for pref in [pref1,pref2,pref3]:
                if prefToCount.has_key(pref):
                    prefToCount[pref] += 1
                else:
                    prefToCount[pref] = 1
        sorted_suffToCount = sorted(suffToCount.items(), key=operator.itemgetter(1),reverse=True)
        sorted_prefToCount = sorted(prefToCount.items(), key=operator.itemgetter(1),reverse=True)
        f = open(outfileSuff,'w');
        f.write("\n".join('%s %s' % x for x in sorted_suffToCount))
        f.close()
        f = open(outfilePref,'w');
        f.write("\n".join('%s %s' % x for x in sorted_prefToCount))
        f.close()

# wordfile = "../data/sec2-21/small.words";
# tagfile = "../data/sec2-21/small.pos";
wordfile = "../data/sec2-21/sec2-21.words";
tagfile = "../data/sec2-21/sec2-21.pos";
lamda = 0.5;
#featureLevel = 1; # basic
featureLevel = 2; # med
#featureLevel = 4; # advanced
numSentences = 5000;
basicFeaturesMinWordCount = 10;
medFeaturesUniCount = 800
medFeaturesBiCount = 800
medFeaturesTriCount = 400
verbose = True

print "initializing..."
t1 = time.clock()
model = MEMMModel.MEMMModel(verbose,basicFeaturesMinWordCount,medFeaturesUniCount,medFeaturesBiCount,medFeaturesTriCount);
model.initModelFromFile(wordfile,tagfile,lamda,featureLevel,numSentences)
#calcStat(model.dictionary,"train_suff_stats.txt","train_pref_stats.txt")
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


#winsound.PlaySound("../yofi_sehel.wav",winsound.SND_FILENAME)