import random

def loadRestaurantData(subset='name', filename='data/restaurant/fz.arff'):  # subset can be 'name' or 'address'
    with open(filename, 'r') as h:
        inData = False
        for l in h.readlines():
            l = l.strip()
            if l == '@data':
                inData = True
                continue
            if inData:
                a = l.split('", "')
                assert(len(a) == 5)
                string = a[0][1:] if subset == 'name' else a[1]
                label  = a[4].split("'")
                label  = label[-2]
                yield (int(label), string)

def makeTwoFolds(dataset):
    label2strings = {}
    for label,string in dataset:
        if label in label2strings: label2strings[label].append(string)
        else: label2strings[label] = [string]
    folds = [[],[]]
    for label,strings in label2strings.iteritems():
        fold = random.randint(0,1)
        folds[fold].append(strings)
    return tuple(folds)

def foldToLabeledPairs(fold, posNegRatio=1.0):
    numPos = 0.
    for strings in fold:
        for i in range(len(strings)):
            for j in range(i):
                yield ((strings[i], strings[j]), 1)
                yield ((strings[j], strings[i]), 1)
                numPos += 2
    for numNeg in range(max(1, int(0.5 + numPos * posNegRatio))):
        i = random.randint(0, len(fold)-1)
        j = random.randint(0, len(fold)-2)
        if i == j: j += 1
        ii = random.randint(0, len(fold[i])-1)
        jj = random.randint(0, len(fold[j])-1)
        yield ((fold[i][ii], fold[j][jj]), 0)

