import autograd.numpy as np
import lols

def sigmoid(z): return 0.5 * (np.tanh(z / 2.) + 1.)

class WeightsParser(object):
    """A helper class to index into a parameter vector. From autograd."""
    def __init__(self):
        self.idxs_and_shapes = {}
        self.num_weights = 0

    def add_shape(self, name, shape):
        if name in self.idxs_and_shapes:
            if shape != self.idxs_and_shapes[name][1]:
                raise Exception("re-adding shape with same name (%s) with different shape (%s vs %s)" % (name, shape, self.idxs_and_shapes[name][1]))
            return
        start = self.num_weights
        self.num_weights += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.num_weights), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)
        
class InOut:
    def _apply_nonlinearity(self, nonlin, res):
        if nonlin == 'none' or nonlin == 'linear':
            return res
        if nonlin == 'relu':
            return 0.5 * (res + np.abs(res))
        if nonlin == 'tanh':
            return np.tanh(res)
        if nonlin == 'softmax':
            res -= res.max()
            res = np.exp(res)
            return res / res.sum()
        raise Exception('unknown nonlinearity: "%s"' % nonlin)
        
    def dots(self, A, bias, nonlin, *args):
        res = lols.dots(A, *args)
        if bias is not None:
            res += 0.1 * bias
        return self._apply_nonlinearity(nonlin, res)

class Constant(InOut):
    def __init__(self, parser, name, dims):
        self.parser = parser
        self.name   = name
        self.dims   = dims
        self.parser.add_shape(name, dims)

    def __call__(self, weights):
        return self.parser.get(weights, self.name)

class FullyConnected(InOut):
    def __init__(self, parser, name, dims, biased=True, nonlin='relu'):
        self.parser = parser
        self.name   = name
        self.dims   = dims
        self.biased = biased
        self.nonlin = nonlin
        self.parser.add_shape(name, dims)
        if biased:
            self.parser.add_shape(name + '_bias', (dims[0]))

    def __call__(self, weights):
        W = self.parser.get(weights, self.name)
        b = self.parser.get(weights, self.name + '_bias') if self.biased else None
        def me(*args): return self.dots(W, b, self.nonlin, *args)
        return me

class GRU(InOut):
    def __init__(self, parser, name, dims):
        self.parser = parser
        self.name   = name
        self.dims   = dims
        self.D_in, self.D_out = dims
        self.parser.add_shape(name + '_W' , (self.D_in , self.D_out))
        self.parser.add_shape(name + '_Wz', (self.D_in , self.D_out))
        self.parser.add_shape(name + '_Wr', (self.D_in , self.D_out))
        self.parser.add_shape(name + '_U' , (self.D_out, self.D_out))
        self.parser.add_shape(name + '_Uz', (self.D_out, self.D_out))
        self.parser.add_shape(name + '_Ur', (self.D_out, self.D_out))

    def __call__(self, weights):
        # input is h_prev and x
        # r     = sigmoid( Wr x + Ur h_prev )
        # z     = sigmoid( Wz x + Uz h_prev )
        # h_new = tanh ( W  x + r . (U h_prev) )
        # h     = (1-z).h_prev + z.h_new   (. = component-wise product)
        # so shapes:
        #   h_prev is D_out, x in D_in, h is D_out, h_new is D_out, z is D_out, r is D_out
        #   W, Wz, Wr : D_in -> D_out
        #   U, Uz, Ur : D_out -> D_out
        W  = self.parser.get(weights, self.name + '_W')
        Wr = self.parser.get(weights, self.name + '_Wr')
        Wz = self.parser.get(weights, self.name + '_Wz')
        U  = self.parser.get(weights, self.name + '_U')
        Ur = self.parser.get(weights, self.name + '_Ur')
        Uz = self.parser.get(weights, self.name + '_Uz')

        def me(h_prev, x):
            r     = sigmoid( np.dot(x, Wr) + np.dot(h_prev, Ur) )
            z     = sigmoid( np.dot(x, Wz) + np.dot(h_prev, Uz) )
            h_new = np.tanh( np.dot(x, W ) + r * np.dot(h_prev, U) )
            h     = (1-z) * h_prev + z * h_new
            return h
        return me
        
    
class Pairwise(InOut):
    def __init__(self, parser, name, dims, nonlin='softmax'):
        self.parser = parser
        self.name   = name
        self.dims   = dims
        self.nonlin = nonlin
        self.parser.add_shape(name, dims)

    def __call__(self, weights):
        W = self.parser.get(weights, self.name)
        def me(x,y): return self._apply_nonlinearity(self.nonlin, np.dot(np.dot(x, W), y))
        return me
    
class Compose(InOut):
    def __init__(self, f, g):
        self.f = f
        self.g = g

    def __call__(self, weights):
        f = self.f(weights)
        g = self.g(weights)
        def me(*args): return f(g(*args))
        return me

class Dropout(InOut):
    def __init__(self, p_drop=0.5):
        self.p_drop = p_drop

    def __call__(self, weights):
        def me(A): return A * (np.random.rand(*A.shape) > self.p_drop)
        return me
        
    
def dropout(p, inOut): return Compose(Dropout(p), inOut)
