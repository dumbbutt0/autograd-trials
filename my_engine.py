# %%
import math
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline



#double underscores (__example__) usually represent instrunctions 
# that python does not currently understand 
class Value:
    def __init__(self,data, _children=(), _op='',label=''):
        self.data =data
        self.grad =0.0
        self._backward= lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
# the purpose of __repr__ is so we ca make clean expressions like a+b
    def __repr__(self):
        return f"Value(data={self.data})"


    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data+other.data,(self,other), '+')
        def _backward():
            self.grad+= 1.0*out.grad
            other.grad+= 1.0*out.grad
        out._backward =_backward
        return out
    
    def __neg__(self):
        return self *-1

    def __sub__ (self, other):
        return self+(-other)


    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other  )
        out = Value(self.data*other.data, (self,other),'*')
        def _backward():
            self.grad+=other.data*out.grad
            other.grad+=self.data*out.grad
        out._backward=_backward
        return out

    def __pow__(self,other):
        assert isinstance(other, (int,float)), "only for int/float powers"
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += other * (self.data ** (other-1)) * out.grad
        out._backward = _backward
        return out

    def __rmul__(self,other):
        return self*other

    def __radd__(self, other):
        return self + other

    def __truediv__(self,other):
        return self * other**-1

    def tanh(self):
        x= self.data
        t=(math.exp(2*x)-1)/(math.exp(2*x)+1)
        out =Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1-t**2)* out.grad
        out._backward= _backward
        return out
        

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,),'exp')
        def _backward():
            self.grad += out.data* out.grad
        out._backward = _backward
        return out

    
    def backward(self):
        # Call ALL nodes in topological order
        topo =[]
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad=1.0 
        for node in reversed(topo):            
            node._backward()



# %%

from graphviz import Digraph

def trace(root):
    #build a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child,v))
                build(child)
    build(root)
    return nodes,edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir':'TB'}) #LR = from left to right

    nodes,edges =trace(root)
    for n in nodes:
        uid =str(id(n))

        #for any value in the graph create a record/node for it
        dot.node(name=uid, label ="{ %s data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            # if this value is the result of an operation, create an op node for it
            dot.node(name = uid + n._op, label = n._op)
            #and connect to this node
            dot.edge(uid + n._op, uid)
    for n1,n2 in edges:
        #connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

#draw_dot(L)

# %%
#inputs x1,x2
x1=Value(2.0, label="x1")
x2=Value(0.0, label='x2')
#weights
w1=Value(-3.0,label='w1')
w2=Value(1.0, label='w2')
#bias of neuron?
b=Value(6.881373587870195432, label='b')

#x1w1 + x2w2 *b
x1w1=x1*w1; x1w1.label='x1w1'
x2w2=x2*w2; x2w2.label='x2w2'
x1w1x2w2=x1w1+x2w2; x1w1x2w2.label='x1w1x2w2'
n= x1w1x2w2 +b ; n.label='n'
#-----
e = (2*n).exp()
o =  (e-1)/(e+1)
#-----
o.label='o'
o.backward()
draw_dot(o)
# %%




# %%
import torch
import random


#leaf nodes(A.K.A inputs to network) need to be expicitly told to require gradients
x1 =torch.Tensor([2.0]).double()  ;x1.requires_grad=True
x2 =torch.Tensor([0.0]).double()  ;x2.requires_grad=True
w1 =torch.Tensor([-3.0]).double()  ;w1.requires_grad=True
w2 =torch.Tensor([1.0]).double()  ;w2.requires_grad=True
b = torch.Tensor([6.881373587870195432]).double() ;b.requires_grad=True
n = x1*w1 +x2*w2 +b
o = torch.tanh(n)

print(o.data.item())
o.backward()

print('----')
print('x2', x2.grad.item())
print('w2', w2.grad.item())
print('x1', x1.grad.item())
print('w1', w1.grad.item())
torch.Tensor([[1,2,3],[4,5,6]])
o.item()
#torch.Size([2,3])
x2.grad.item()
#Python uses double float precision therefore .double() is used because tensors default type is 
#float32 and we want float64 for python torch.Tensor([2.0]).double().dtype == torch.float64

# %%
class Neuron:
    def __init__(self,nin):
        #w = weights
        self.w =[Value(random.uniform(-1,1)) for _ in range (nin)]
        #b = bias
        self.b = Value(random.uniform(-1,1))
    def __call__(self,x):
        #w * x + b
        act= sum((wi*xi for wi,xi in zip(self.w,x)),self.b)
        out = act.tanh()
        return out
    def parameters(self):
        return self.w + [self.b]

x= [2.0,3.0]
n= Neuron(2)
n(x)



class Layer:
    def __init__(self,nin,nout):
        self.neurons =[Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs =[n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        #these are identical
        #params = []
        #for neuron in self.neurons:
        #    ps= neuron.parameters()
        #    params.extemd(ps)
        #return params

x= [2.0,3.0]
n= Layer(2, 3)
n(x)

class MLP:
    def __init__(self,nin,nouts):
        sz =[nin] + nouts
        self.layers = [Layer(sz[i], sz [i+1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return(x)
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

x= [2.0,3.0, -1.0]
n= MLP(3,[4, 4, 1])
n(x)
draw_dot(n(x))
# %%
xs = [
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] #desired targets

for k in range(10):
    #forward pass
    ypred =[n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys,ypred))

#all the gradients stack up if we dont 0 them out
    #backwards pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    #update
    for p in n.parameters():
        p.data += -0.01  * p.grad
    print(k, loss.data)

ypredict =[n(x) for x in xs]
ypredict

# for ygroundtruth and youtput in zip of ys in ypredict
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypredict))
#higher loss output means we are further from ground truth (we want 0.0 if possible)
loss
# %%
ypredict =[n(x) for x in xs]
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypredict))

loss.backward()
loss
draw_dot(loss)
'''
neural nets are multi-layer perceptrons represented as mathematical functions that take input as data, weights and parameters

there is a mathematical expression for a forward pass followed by a loss function that measures the accuracy of
the predictions and will have a lower loss when the predictions are closer to the ground truth while aiming for 0.0
(manipulate loss accordingly to make the neural net act in the right direction)
we then back propogate for the gradient and tune parameters to reach a more accurate loss output. we do this continuosly in order to achieve the gradient descent
and minimize the loss as much as possible to be as accurate as possible

'''
# %%
##
# things got messy so i moved to the clean engine
##
