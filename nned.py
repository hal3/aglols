import autograd.numpy as np
import Levenshtein
import random
from agnn import *
import lols
import datasets
import canopy
import sys
#np.random.seed(90210)

def onehot(V, i):
    A = np.zeros(V)
    if i is not None: A[i] = 1.
    return A

class AlwaysSubstituteRef(lols.Reference):
    def next(self): return [0]
    def step(self,e): pass

class RandomRef(lols.Reference):
    def next(self): return [0,1,2]
    def step(self,e): pass

class SimpleDistanceRef(lols.Reference):
    def __init__(self, input_output_label):
        (self.A,self.B),z = input_output_label
        self.i = 0
        self.j = 0

    def step(self, e):
        if   e == 0: self.i,self.j = self.i+1, self.j+1
        elif e == 1:        self.j =           self.j+1
        elif e == 2: self.i        = self.i+1

    def next(self):
        A,B,i,j = self.A, self.B, self.i, self.j
        if A[i] == B[j]: return [0] # substitute
        elif len(A) - i == len(B) - j: return [0] # substitute
        elif len(A) - i <  len(B) - j: return [1] # advance j
        else: return [2]

class EditDistanceRef(lols.Reference):
    def __init__(self, input_output_label):
        (self.A,self.B),z = input_output_label
        self.i = 0
        self.j = 0

    def step(self, e):
        if   e == 0: self.i,self.j = self.i+1, self.j+1
        elif e == 1:        self.j =           self.j+1
        elif e == 2: self.i        = self.i+1

    def next(self):
        i,j = self.i, self.j
        A,B = self.A[i:], self.B[j:]
        ops = Levenshtein.editops(A,B)
        if len(ops) == 0: return [0]  # substitute
        (op,ii,jj) = ops[0]
        if ii > 0 or jj > 0: return [0] # equal / substitute
        if op == 'replace': return [0]
        if op == 'insert':  return [1]
        if op == 'delete':  return [2]
        raise Exception('unknown op %s in edit from %s to %s' % op, A, B)

                            
class NNED:
    def __init__(self, mkRef, cfg):
        self.cfg = cfg
        self.mkRef = mkRef

        self.vocab_size  = len(cfg['vocab'])  # assume 0 == OOV, 1 == </s>
        
        self.parser = WeightsParser()

        #self._backward_gru = GRU(self.parser, 'backward_gru', (vocab_size, cfg['backward_gru']))
        self._make_predict = FullyConnected(self.parser, 'make_predict', (3, cfg['d_hidden']), nonlin='softmax')
        self._make_hidden  = FullyConnected(self.parser, 'make_hidden',  (cfg['d_hidden'], cfg['d_state'] + 2*self.vocab_size + 1), nonlin='softmax')
        self._evolve_state = FullyConnected(self.parser, 'evolve_state', (cfg['d_state'], cfg['d_state'] + 2*self.vocab_size + 1 + 3), nonlin='softmax')
        self._predict_z    = FullyConnected(self.parser, 'predict_z',    (2, cfg['d_state']), nonlin='softmax')
        self._init_state   = Constant(self.parser, 'init_state',   (cfg['d_state']))

        self.num_weights = self.parser.num_weights

    def _run(self, input_output_label, search, weights):
        (A,B),z = input_output_label

        make_predict = self._make_predict(weights)
        make_hidden  = self._make_hidden(weights)
        evolve_state = self._evolve_state(weights)
        predict_z    = self._predict_z(weights)

        i = 0  # position in A
        j = 0  # position in B
        k = 0  # number of edit operations so far
        q = self._init_state(weights)
        edits = []
        ref   = self.mkRef(input_output_label)
        numErr = 0.
        while i < len(A) and j < len(B):
            # current values
            a = 1 if i >= len(A) else self.cfg['vocab'].get(A[i], 0)
            b = 1 if j >= len(B) else self.cfg['vocab'].get(B[j], 0)
            a_eq_b = np.array([1. if i < len(A) and j < len(B) and A[i] == B[j] else -1.])
            
            # compute the hidden representation
            hidden = make_hidden(q, onehot(self.vocab_size, a), onehot(self.vocab_size, b), a_eq_b)

            # there's only a valid prediction to make if _both_ i < |A| and j < |B| otherwise we're forced to ins/del
            if   i >= len(A): e = 1
            elif j >= len(B): e = 2
            else:              # we have our 3-way choice
                pred = make_predict(hidden)
                e    = search.predict(ref.next(), pred)
#                e = 0

            # we've got an action to take; 0 == substitute, 1 == skip a, 2 == skip b
            if   e == 0: i,j = i+1, j+1
            elif e == 1:   j =      j+1
            elif e == 2: i   = i+1

            k += 1
            edits.append(e)
            ref.step(e)
            q  = evolve_state(q,
                             onehot(self.vocab_size, a),
                             onehot(self.vocab_size, b),
                             a_eq_b,
                             onehot(3, e))
            pz = predict_z(q)
            zhat = search.predict([z], pz)
            if z is not None and (zhat == 1) != z:
                #if z: numErr += 1.
                #else: numErr += 0.1
                numErr += 1.
                #break

#        q  = evolve_state(q,
#                          onehot(self.vocab_size, a),
#                          onehot(self.vocab_size, b),
#                          a_eq_b,
#                          onehot(3, e))
#        pz = predict_z(q)
#        zhat = search.predict([z], pz)   # don't actually care what happens, just need the loss!

        if z is not None and (zhat == 1) != z: numErr += 10 # lots of mistake at the end
        #if z is not None: search.declareLoss( 1. if (zhat == 1) != z else 0. )
        if z is not None: search.declareLoss(numErr)
                
        #print edits,q,pz,zhat,z
        #print q,pz
        return zhat

def make_character_vocabulary(data):
    vocab = { '**OOV**' : 0, '</w>': 1 }
    for (s1,s2),lab in data:
        for c in s1 + s2:
            if c not in vocab: vocab[c] = len(vocab)
    return vocab
        
def div(x,y): return 0. if x == 0 or y == 0 else (float(x) / float(y))

def pairwiseFScore(labels, a, b):
    predLabels = {}
    predictedOn = set()
    for ((s1,s2),_foo),z in zip(a,b):  # s1,s2 is string pair, z is prediction if they are the same or not
        #print s1,s2,_foo,z
        predictedOn.add(s1)
        predictedOn.add(s2)
        if z == 0: continue # not linked
        #print '%s == %s' % (s1,s2)
        if s1 in predLabels and s2 in predLabels:
            l1 = predLabels[s1]
            l2 = predLabels[s2]
            if l1 != l2:
                for s3,l3 in predLabels.items():
                    if l3 == l2: predLabels[s3] = l1
        elif s1 in predLabels:
            predLabels[s2] = predLabels[s1]
        elif s2 in predLabels:
            predLabels[s1] = predLabels[s2]
        else:
            predLabels[s1] = len(predLabels)
            predLabels[s2] = predLabels[s1]
    #print predLabels
    I = 0
    T = 0
    P = 0
    for s1 in predictedOn:
        l1 = labels[s1]
        for s2 in predictedOn:
            l2 = labels[s2]
            if s2 <= s1: continue
            if l1 == l2:
                T += 1
                if s1 in predLabels and s2 in predLabels and predLabels[s1] == predLabels[s2]:
                    I += 1
    for s1,l1 in predLabels.iteritems():
        for s2,l2 in predLabels.iteritems():
            if s2 <= s1: continue
            if l1 == l2:
                P += 1
    pre = div(I, P)
    rec = div(I, T)
    fsc = div(2 * pre * rec, pre + rec)
    print >>sys.stderr, 'I=%d T=%d P=%d len(a)=%d pre=%g rec=%g fsc=%g\t' % (I,T,P,len(a),pre,rec,fsc),
    return fsc
        
    
def evaluate(labels,a,b):
    res = [(z==1, zhat==1) for (_,z),zhat in zip(a,b)]
    N   = len(res)
    if N == 0: return 0., 0.
    acc = div(sum([t==s for t,s in res]) , float(N))
    pre = div(sum([t and s for t,s in res]) , sum([s for _,s in res]))
    rec = div(sum([t and s for t,s in res]) , sum([t for t,_ in res]))
    fsc = div(2 * pre * rec , (pre + rec))
    return fsc, pairwiseFScore(labels,a,b), acc
    
#test_data = [(('ac', 'ac'), 1), (('ac', 'ca'), 0)]
#vocab     = { '':0, '$': 1, 'a': 2, 'b': 3, 'c': 4 }

def buildTestCanopies(fold, labels, includeSelf):
    II = canopy.InvIndex()
    for strings in fold:
        label = len(labels)
        for s in strings:
            II.add(s)
            labels[s] = label
    canopies = list(canopy.buildCanopies(II, 0.5, 0.0, Levenshtein.jaro))
    string2canopy = {}
    for n,strings in enumerate(canopies):
        for string in strings:
            if string in string2canopy: string2canopy[string].add(n)
            else: string2canopy[string] = set([n])
    #return II, canopy, string2canopy
    numMiss = 0.
    numTot  = 0.
    for strings in fold:
        for i in range(len(strings)):
            s1 = strings[i]
            c1 = string2canopy[s1]
            for j in range(i):
                s2 = strings[j]
                c2 = string2canopy[s2]
                if len(c1 & c2) == 0:
                    numMiss += 1
                    print >>sys.stderr, 'miss: "%s" / "%s" (%g)' % (s1, s2, 1. - Levenshtein.jaro(s1, s2))
                numTot += 1    
    print >>sys.stderr, 'missing total of %g / %g' % (numMiss, numTot)
    print >>sys.stderr, 'largest canopy %d' % max(map(len,canopies))
    pairs = []
    for string_set in canopies:
        strings = list(string_set)
        for i in range(len(strings)):
            for j in range(i+includeSelf):
                s1,s2 = strings[i], strings[j]
                if random.random() < 0.5: s1,s2 = s2,s1
                pairs.append( ((s1,s2), labels[s1] == labels[s2]) )
    return pairs

fold1,fold2 = datasets.makeTwoFolds(datasets.loadRestaurantData(subset='name'))
labels      = {}
train       = buildTestCanopies(fold1, labels, False)
#train       = list(datasets.foldToLabeledPairs(fold1))
test        = buildTestCanopies(fold2, labels, False)
#test        = list(datasets.foldToLabeledPairs(fold2))
vocab       = make_character_vocabulary(train)
random.shuffle(train)
random.shuffle(test)
numData = 1000
print '#train %d, #test %d, %d positive in train' % (len(train), len(test), sum([x for _,x in train[:numData]]))

def option(optName, convert=float):
    for a in sys.argv:
        if a.startswith(optName + '='):
            return convert(a[len(optName)+1:])
    return None

def subsampleTrain():
    pos = [(x,y) for (x,y) in train if y]
    neg = [(x,y) for (x,y) in train if not y]
    random.shuffle(neg)
    total = pos + neg[:len(pos)]
    random.shuffle(total)
    return total

model     = NNED(EditDistanceRef,
                 { 'vocab'        : vocab,
                   'd_hidden'     : option('d_hidden', int) or 50,
                   'd_state'      : option('d_state',  int) or 50,
                   })
weights   = np.random.randn(model.num_weights) * (option('init_variance') or 0.001)

# learner   = lols.DAgger(model,
#                         alpha=lols.NoDecay(1.0))
# lols.sgd(learner,
#          40,
#          train,
#          train,
#          [],
#          weights,
#          batchSize=1,
#          eta0=0.1,
#          adaptive=True,
#          computeLosses=lambda a,b: evaluate(labels, a, b),
#          senseIsMinimize=False,
# #         outputExpDelay=True,
# #         extraObjective=lambda w: 0.0 * np.linalg.norm(w),   # l2 regularizer
#          )

learner   = lols.LOLS(model,
                      alpha=lols.ExponentialDecay(option('alpha') or 0.9),
                      #beta=lols.ExponentialDecay(0.9),
                      beta=lols.NoDecay(option('beta') or 0.5),
                      rollout_method=lols.LOLS.ROLLOUT_MIX_PER_STATE,
                      #rollout_high_entropy_only_count=1,
                      #rollout_subsample_time=0.2,
                      )
lols.sgd(learner,
         option('passes', int) or 50,
         subsampleTrain,
         test,
         [],
         weights,
         batchSize=1,
         eta0=option('eta0') or 0.1,
         adaptive=True,
         computeLosses=lambda a,b: evaluate(labels,a,b),
         senseIsMinimize=False,
#         outputExpDelay=True,
         extraObjective=lambda w: (option('l2') or 0.) * np.linalg.norm(w),   # l2 regularizer
         )
