import autograd.numpy as np
import autograd
import autograd.util
import sys
import scipy.optimize
import random

neginf = float('-inf')

class Reference:
    def __init__(self, y):
        self.y = y

    def step(self, p): raise Exception('step not defined')
    def loss(self):    raise Exception('loss not defined')
    def next(self):    raise Exception('next not defined')

class TestReference(Reference):
    def step(self, p): pass
    def loss(self): pass
    def next(self): pass
        
class Decay:
    pass

class NoDecay(Decay):
    def __init__(self, value): self.value = value
    def __call__(self, T): return self.value
        
class ExponentialDecay(Decay):
    def __init__(self, alpha, lower_bound=0., upper_bound=1.):
        self.alpha = alpha
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
    def __call__(self, T):
        return min(self.upper_bound, max(self.lower_bound, self.alpha ** T))

class LinearDecay(Decay):
    def __init__(self, slope, offset=1., lower_bound=0., upper_bound=1.):
        self.slope = slope
        self.offset = offset
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
    def __call__(self, T):
        return min(self.upper_bound, max(self.lower_bound, self.offset - self.slope * T))

class InverseSigmoidDecay(Decay):
    def __init__(self, kappa, lower_bound=0., upper_bound=1.):
        self.kappa = kappa
    def __call__(self, T):
        return min(self.upper_bound, max(self.lower_bound, self.kappa / (self.kappa + np.exp(T / self.kappa))))

class UserDecay(Decay):
    def __init__(self, f):
        self.f = f
    def __call__(self, T): return self.f(T)

def dots(A, *args):
    # compute dot(A, [args[0] args[1] args[2] ...])
    n,m = A.shape
    tot = 0
    for a in args:
        tot += a.shape[0]
    if tot != m:
        for a in args: print a.shape
        raise Exception('dots: mismatched dimensions. A.shape=%s, total shape=%d' % (str(A.shape), tot))
    return np.dot(A, np.concatenate(list(args)))

def tanh(A, *args):
    return np.tanh(dots(A, *args))

def softmax(A, *args):
    # TODO: better softmax
    if len(args) > 0:
        p = dots(A, *args)
    else:
        p = A
    p -= p.max()
    p = np.exp(p)
    #p = np.exp(dots(A, *args))
    return p / p.sum()


def relu(A, *args):
    p = dots(A, *args)
    return 0.5 * (p + np.abs(p))

#def nonlin(A, *args): return tanh(A, *args)

def entropy(p): return -np.dot(p, np.log(p + 1e-10))

def randomRef(refActions): return np.random.choice(list(refActions))

def maxlikLoss(prob, refActions):
    # find prob of best action
    bestProb = prob[np.array(list(map(int,refActions)))].sum()
    return -np.log(bestProb + 1e-10)

class Learner:
    def __init__(self, model):
        self.model = model
        self.predictRun = False

    def set_weights_copy(self, weights_copy): pass
        
    def train(self, input_output_seq, weights):
        self.totalLoss = 0.
        if isinstance(input_output_seq, tuple): input_output_seq = [input_output_seq]
        for input_output in input_output_seq:
            self.model._run(input_output, self, weights)
        return self.totalLoss # / len(input_output_seq)

    def predict(self, refActions, pred, disallowed=None):
        # default prediction is to just use pred
        r = pred.argmax()
        if disallowed is not None and r == disallowed:
            r = 0
            for i in range(1, len(pred)):
                if i != disallowed and pred[i] > pred[r]: r = i
        return r

    def declareLoss(self, loss): pass

    def run(self, input_output_seq, weights):
        #print "========= BEGIN RUN ================================================="
        oldPredictRun = self.predictRun
        self.predictRun = True
        res = []
        if isinstance(input_output_seq, tuple): input_output_seq = [input_output_seq]
        for input_output in input_output_seq:
            input_only = input_output
            if isinstance(input_only, tuple): input_only = input_only[0]
            res.append(self.model._run((input_only,None), self, weights))
        self.predictRun = oldPredictRun
        #print "========= END RUN ==================================================="
        #print res
        return res
            

class MaxLik(Learner):
    def __init__(self, model):
        Learner.__init__(self, model)

    def train(self, input_output_seq, weights):
        return Learner.train(self, input_output_seq, weights)
    
    def predict(self, refActions, pred, disallowed=None):
        if self.predictRun: return Learner.predict(self, refActions, pred, disallowed)
        # maxlik always returns a reference action, and always accumulates policy loss
        mll  = maxlikLoss(pred, refActions)  # TODO: disallowed?
        self.totalLoss += mll
        #print 'ref =', refActions, '; pred =', pred
        return randomRef(refActions)
        
    def declareLoss(self, loss):
        # totally irrelevant
        return

class DAgger(Learner):
    def __init__(self, model, alpha=ExponentialDecay(0.99)):
        Learner.__init__(self, model)
        self.num_examples = 0.
        self.alpha = alpha

    def train(self, input_output_seq, weights):
        self.num_examples += 1.0
        self.p_ref = self.alpha(self.num_examples)
        self.totalLossCount = 0.
        totalLik = Learner.train(self, input_output_seq, weights)
        #print self.totalLossCount
        return totalLik / self.totalLossCount

    def predict(self, refActions, pred, disallowed=None):
        if self.predictRun: return Learner.predict(self, refActions, pred, disallowed)
        mll  = maxlikLoss(pred, refActions)  # TODO: disallowed
        self.totalLoss += mll
        self.totalLossCount += 1
        if np.random.rand() < self.p_ref: a = randomRef(refActions)
        else:                             a = Learner.predict(self, refActions, pred, disallowed)
        #sys.stderr.write(str(a))
        return a
        
    def declareLoss(self, loss):
        # totally irrelevant
        return

class LOLS(Learner):
    RUN,BACKBONE,LEARN = 0,1,2
    ROLLOUT_REF,ROLLOUT_MIX,ROLLOUT_MIX_PER_STATE = 0,1,2
    
    def __init__(self, model, rollout_method=ROLLOUT_MIX, alpha=ExponentialDecay(0.99), beta=NoDecay(0.5), rollout_subsample_time=1.0, rollout_high_entropy_only_count=None):
        Learner.__init__(self, model)
        self.num_examples = 0.
        self.alpha = alpha
        self.state = LOLS.RUN
        self.rollout_method = rollout_method
        self.beta = beta
        self.rollout_subsample_time = rollout_subsample_time
        self.rollout_high_entropy_only_count = rollout_high_entropy_only_count
        self.weights_copy = None

    def set_weights_copy(self, weights_copy):
        self.weights_copy = weights_copy
        
    def train(self, input_output_seq, weights):
        self.num_examples += 1.0
        self.p_ref = self.alpha(self.num_examples)
        self.weights = weights
        self.total_loss = 0.

        assert(self.weights_copy is not None)
        
        if isinstance(input_output_seq, tuple): input_output_seq = [input_output_seq]
        for input_output in input_output_seq:
            self.train_one(input_output)
        
        self.state = LOLS.RUN
        return self.total_loss

    def train_one(self, input_output):
        self.state = LOLS.BACKBONE
        self.branches = []
        self.backbone = []
        self.t = 0
        self.model._run(input_output, self, self.weights_copy)
        #print 'backbone = %s' % (str(self.backbone))

        if   self.rollout_method == LOLS.ROLLOUT_REF:
            self.rollout_p_ref = 1.0
        elif self.rollout_method == LOLS.ROLLOUT_MIX:
            self.rollout_p_ref = 1.0 if np.random.rand() < self.beta(self.num_examples) else 0.

        timesteps = None
        if self.rollout_high_entropy_only_count is not None:
            timesteps = [(h, t) for t,(h,_) in enumerate(self.backbone)]
            timesteps.sort(reverse=True)
#            print timesteps
            timesteps = set([t for _,t in timesteps[:self.rollout_high_entropy_only_count]])
            
        self.state = LOLS.LEARN
        for self.t0 in range(len(self.backbone)):
            if np.random.rand() > self.rollout_subsample_time: continue
            if timesteps is not None and self.t0 not in timesteps: continue
            
            if self.rollout_method == LOLS.ROLLOUT_MIX_PER_STATE:
                self.rollout_p_ref = 1.0 if np.random.rand() < self.beta(self.num_examples) else 0.
            #self.losses = []
            self.losses0= []
            self.preds  = []
            for self.a0 in self.branches[self.t0]:
                self.t = 0
                #print 't0 %d, a0 %d' % (self.t0,self.a0)
                self.model._run(input_output, self, self.weights)
            self.losses = self.losses0
            best_loss = min(self.losses)
            #print 'losses =', self.losses, ' best_loss =', best_loss
            #print 'losses0=', self.losses0
            assert(len(self.losses) == len(self.preds))
            this_loss = 0.
            best_min_loss = None
            #print self.losses
            for l,p in zip(self.losses, self.preds):
                if l <= best_loss:
                    if best_min_loss is None or p > best_min_loss:
                        best_min_loss = p
                else:  #l > best_loss:
                    this_loss += (l - best_loss) * np.log(1. - p)
            self.total_loss -= this_loss
            #this_loss = - np.log(this_loss)
            #for l,p in zip(self.losses, self.preds):
            #    if l <= best_loss:
            #        this_loss += np.log(p)
            #this_loss += np.log(best_min_loss)
            #self.total_loss -= np.log(best_min_loss + 1e-10)  # this_loss
                #self.total_loss += (p - l) * (p - l)   # TODO: is this right????
                #self.total_loss += (1 + l - best_loss) * np.log(p)
                
                

    def predict(self, refActions, pred, disallowed=None):
        res = self.predict0(refActions, pred, disallowed)
        if self.state == LOLS.BACKBONE:
            self.backbone.append( (entropy(pred) / len(pred), res) )
        #print 't=%d ref=%s predicting %d\t\tpred=%s' % (self.t-1,str(refActions),res,str(pred))
        return res
    
    def predict0(self, refActions, pred, disallowed=None):
        # if we're in "run" mode, just make a prediction
        if self.predictRun: return Learner.predict(self, refActions, pred, disallowed)
            
        res = None
        p_ref = self.alpha(self.num_examples)  # default roll-in
        if self.state == LOLS.BACKBONE:
            #self.branch_factor = max(self.branch_factor, len(pred))
            self.branches.append( [i for i in range(len(pred)) if i != disallowed] )
            pass # everything is set up properly
        elif self.state == LOLS.LEARN:
            if self.t < self.t0:
                res = self.backbone[self.t][1]
            elif self.t == self.t0:
                res = self.a0
                #mll = maxlikLoss(pred, set([self.a0])) #refActions)  # TODO: disallowed
                #print pred
                p = pred[self.a0]
                self.preds.append(p)
                #self.losses.append(0.0 if self.a0 in refActions else 1.0)
            else: # t > t0
                p_ref = self.rollout_p_ref
            self.t += 1

        if res is not None:
            return res

        if np.random.rand() < p_ref:
            return randomRef(refActions)
        else:
            return Learner.predict(self, refActions, pred, disallowed)
        
    def declareLoss(self, loss):
        if self.state == LOLS.LEARN:
            #print 'declareLoss(%g)' % loss
            self.losses0.append(loss)

globalEpoch = 0
globalObjective = 0
globalBestWeights = None
def value_and_grad(learner, trainingData, weights, extraObjective=None):
    def trainIt(weights):
        global globalObjective
        globalObjective += learner.train(trainingData, weights)
        if extraObjective is not None:
            globalObjective += extraObjective(weights)
        return globalObjective
    return autograd.value_and_grad(trainIt)
                    
def makePrintUpdate(learner, mkTrainingData, devData, testData, computeLosses, targetDict=None, senseIsMinimize=True):
    #allTrainingItems = set([spellOut(targetDict, x[1][:-1]) for x in trainingData])
    def printUpdate(weights, overrideEpoch=None):
        global globalObjective, globalEpoch, globalBestWeights
        if overrideEpoch is None:
            trainingData = mkTrainingData()
            trainingPredictions = learner.run(trainingData, weights)
            trLosses = (0,0) if computeLosses is None else computeLosses(trainingData, trainingPredictions)
            globalEpoch += 1
        else:
            trLosses = 0,0
        devPredictions  = learner.run(devData, weights)
        testPredictions = learner.run(testData, weights)
        deLosses = (0,0) if computeLosses is None else computeLosses(devData, devPredictions)
        teLosses = (0,0) if computeLosses is None else computeLosses(testData, testPredictions)
        #print trainingPredictions
        isBest = ''
        #if trLosses[1] == 0: raise Exception('done on epoch %d!' % (globalEpoch))
        obj = globalObjective.value if isinstance(globalObjective,autograd.core.FloatNode) else globalObjective
        if globalBestWeights is None or ((senseIsMinimize and deLosses[1] < globalBestWeights[0][3][1]) or ((not senseIsMinimize) and deLosses[1] > globalBestWeights[0][3][1])):
            globalBestWeights = ((obj, globalEpoch, trLosses, deLosses, teLosses), weights.copy())
            isBest = '\t*'
        if globalEpoch % 1 == 0 or isBest != '':
            print "%s %d\tobj %g\ttr %s\tde %s\tte %s%s" % ('epoch' if overrideEpoch is None else 'examp', globalEpoch if overrideEpoch is None else overrideEpoch, obj, trLosses, deLosses, teLosses, isBest)
            globalObjective = 0.
            devList = range(len(devData))
            random.shuffle(devList)
            devList = sorted(devList[:20])
            #for n in devList:
            #    devPred = spellOut(targetDict, devPredictions[n][:-1])
            #    print '[dev %d ref] %s' % (n, spellOut(targetDict, devData[n][1][:-1]))
            #    print '[dev %d sys] %s\t%s' % (n, devPred, '*' if devPred in allTrainingItems else '')
            
    return printUpdate

def gd(learner, numEpochs, trainingData, devData, testData, weights, computeLosses=None, outputFrequency=1, eta0=0.1, initial_t=0, power_t=0.5, extraObjective=None, senseIsMinimize=True):
    global globalEpoch, globalBestWeights
    globalEpoch,globalBestWeights = 0, None
    learner.set_weights_copy(weights.copy())  # TODO: this won't work :(
    obj_and_grad = value_and_grad(learner, trainingData, weights, extraObjective)
    printUpdate  = makePrintUpdate(learner, trainingData, devData, testData, computeLosses, senseIsMinimize)
    for epoch in range(1, numEpochs+1):
        _, gradient = obj_and_grad(weights)
        eta = eta0 / (1 if power_t == 1 else ((epoch + initial_t) ** power_t))
        weights -= eta * gradient
        if epoch % outputFrequency == 0:
            printUpdate(weights)

    print 'best =', globalBestWeights[0]
    return globalBestWeights

def log2(v):
    if v <= 0: return 0
    return int(np.log(v) / np.log(2))

def sgd(learner, numEpochs, mkTrainingData, devData, testData, weights, computeLosses=None, batchSize=1, outputFrequency=1, outputExpDelay=False, eta0=0.01, initial_t=0, power_t=0.5, extraObjective=None, adaptive=False, clipping=False, targetDict=None, senseIsMinimize=True):
    global globalEpoch, globalBestWeights
    globalEpoch,globalBestWeights = 0, None
    printUpdate  = makePrintUpdate(learner, mkTrainingData, devData, testData, computeLosses, targetDict=targetDict, senseIsMinimize=senseIsMinimize)
    globalEpoch = 0
    sum_grad_squared = None
    totalExamples = 0
    for epoch in range(1, numEpochs+1):
        trainingData = mkTrainingData()
        for start in range(0, len(trainingData), batchSize):
            data = trainingData[start:start+batchSize]
        
            learner.set_weights_copy(weights.copy())
            obj_and_grad = value_and_grad(learner, data, weights, extraObjective)
            _, gradient = obj_and_grad(weights)
            eta = eta0 / (1 if power_t == 0 else ((epoch + initial_t) ** power_t))
            gradient[np.isnan(gradient)] = 0
            gradient[np.isinf(gradient)] = 0
            gradient *= eta
            if clipping:
                numBig = sum(gradient < -1) + sum(gradient > 1)
                if numBig > 0:
                    print 'clipping %d / %d gradient terms, avg|grad| %g' % (numBig, len(gradient), np.mean(np.abs(gradient)))
                    gradient[gradient > 1] = 1
                    gradient[gradient < -1] = -1
            if adaptive:
                if sum_grad_squared is None:
                    sum_grad_squared = 1e-4 + gradient * gradient
                else:
                    gradient /= np.sqrt(sum_grad_squared)
                    sum_grad_squared += gradient * gradient
            weights -= gradient
            if outputExpDelay and log2(totalExamples) != log2(totalExamples+len(data)):
                printUpdate(weights, totalExamples)
            totalExamples += len(data)
        if epoch % outputFrequency == 0:
            printUpdate(weights)
    return globalBestWeights

    

def spminimize(learner, numEpochs, trainingData, devData, testData, weights, computeLosses=None, outputFrequency=1, method='CG', jac=True, options={}, extraObjective=None, senseIsMinimize=True):
    global globalEpoch, globalBestWeights
    globalEpoch,globalBestWeights = 0, None
    obj_and_grad = value_and_grad(learner, trainingData, weights, extraObjective)

    options2 = options
    options2['maxiter'] = numEpochs
    options2['gtol']    = 1e-10
    options2['disp']    = True
    globalEpoch = 0
    printUpdate = makePrintUpdate(learner, trainingData, devData, testData, computeLosses, senseIsMinimize=senseIsMinimize)
    learner.set_weights_copy(weights.copy())  # TODO: this won't work :(
    
    result = scipy.optimize.minimize(obj_and_grad, weights,
                                     jac=jac, method=method, options=options2,
                                     tol=0,
                                     )#callback=printUpdate)
    return globalBestWeights
    
