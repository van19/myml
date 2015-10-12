# -*- coding: utf-8 -*-
import math
import random
import sys
import copy
def sigmoid(x):
    #sys.stderr.write("%f\n"%x)
    if x<0:
        ex = math.exp(x)
        return ex/(ex+1)
    return 1/(1+math.exp(-x))
    
def sigmoid_grad(x):    
    s = sigmoid(x)
    return s*(1-s)
def gussian(mu,sigma):
    r1 = random.random()
    r2 = random.random()
    r = mu + sigma * math.sqrt(-2*math.log(r1)) / math.cos(2*math.pi*r2)
    return r
def mean(x):
    return (random.random()*2-1)*x

class Sample:
    def __init__(self):
        self.vec={}
        self.label = []
    def vec_pos(self,x):
        if self.vec.has_key(x):
            return self.vec[x]
        else:
            return 0

def gen_sample(line):
    sl=line.strip().split(" ")
    res=Sample()
    begin=0
    if sl[0].strip().find(":")<0:
        label_list = sl[0].strip().split(",")
        for i in label_list:
            res.label.append(float(i))
        begin=1
    res.vec[0]=1
    for i in xrange(begin,len(sl)):
        ssl=sl[i].strip().split(":")
        try:
            k=int(ssl[0].strip())
            v=float(ssl[1].strip())
            res.vec[k] = v
        except:
            print line
    return res
class Node:
    def __init__(self):
        self.z = 0.0
        self.sigma = 0.0
        self.w = []
    def get_z(self):
        return self.z
    def set_z(self,v):
        self.z = v
    def get_sigma(self):
        return self.sigma
    def set_sigma(self,v):
        self.sigma = v
class Edge:
    def __init__(self):
        self.value = 0.0
        self.delta = 0.0
    def get_value(self):
        return self.value
    def set_value(self,v):
        self.value = v
    def get_delta(self):
        return self.delta
    def set_delta(self,v):
        self.delta = v

class NeuralNetwork:
    def __init__(self):
        self.alpha = 0.01
        self.lam = 0.0
        self.layer = []
    def active(self,l,i):
        z = self.get_z(l,i)
        if self.is_input_layer(l):
            return z
        if self.is_bias_node(l,i):
            return z
        return sigmoid(z)
    def gradient(self,l,i):
        if self.is_input_layer(l):
            return 1.0
        if self.is_bias_node(l,i):
            return 1.0
        z = self.get_z(l,i)
        return sigmoid_grad(z)
    def is_input_layer(self,l):
        if l == 0:
            return True
        return False
    def is_output_layer(self,l):
        if l == len(self.layer) - 1:
            return True
        return False
    def is_bias_node(self,l,i):
        if not self.is_output_layer(l):
            if i == 0:
                return True
        return False
    def get_z(self,l,i):
        return self.layer[l][i].get_z()
    def set_z(self,l,i,v):
        self.layer[l][i].set_z(v)
    def get_sigma(self,l,i):
        return self.layer[l][i].get_sigma()
    def set_sigma(self,l,i,v):
        self.layer[l][i].set_sigma(v)
    def get_w(self,l,i,j):
        n = j - 1
        #下一层是输出层则没有偏置节点,因此w[0]不是b
        if self.is_output_layer(l+1):
            n = j
        #sys.stderr.write("get_w:%d,%d,%d\n"%(l,i,n))
        return self.layer[l][i].w[n].get_value()
    def set_w(self,l,i,j,value):
        n = j - 1
        if self.is_output_layer(l+1):
            n = j
        self.layer[l][i].w[n].set_value(value)
    def get_delta(self,l,i,j):
        n = j - 1
        #下一层是输出层则没有偏置节点,因此w[0]不是b
        if self.is_output_layer(l+1):
            n = j
        return self.layer[l][i].w[n].get_delta()
    def set_delta(self,l,i,j,value):
        n = j - 1
        if self.is_output_layer(l+1):
            n = j
        self.layer[l][i].w[n].set_delta(value)
    def update_delta(self,l,i,j,value):
        n = j - 1
        if self.is_output_layer(l+1):
            n = j
        delta = self.layer[l][i].w[n].get_delta()
        self.layer[l][i].w[n].set_delta(delta+value)

    def make_nn(self,a):
        self.layer = []
        #a = [4,3,2,1]
        sz = len(a)
        for i in xrange(sz):
            layer = []
            for j in xrange(a[i]):
                node = Node()
                if j == 0 and i != sz-1:
                    node.z = 1.0;
                layer.append(node)
            self.layer.append(layer)    
        self.init_weight()
    def init_weight(self):
        layer_num = len(self.layer)
        for i in xrange(layer_num - 1):
            node_num = len(self.layer[i])
            next_node_num = len(self.layer[i+1])
            for j in xrange(0, node_num):
                nt = self.layer[i][j]
                begin = 1
                if self.is_output_layer(i+1):
                    begin = 0
                for k in xrange(begin,next_node_num):
                    #r = gussian(1,0.01)
                    r = 0.2+mean(0.1)
                    edge = Edge()
                    edge.value = r
                    nt.w.append(edge)
    def clear_nn_delta(self):
        for l in xrange(len(self.layer) - 1):
            begin = 1
            if self.is_output_layer(l+1):
                begin = 0
            for i in xrange(len(self.layer[l])):
                for j in xrange(begin, len(self.layer[l+1])):
                    self.set_delta(l,i,j,0.0)
    def forward_update_node(self,l,i):
        if l <= 0 or l >= len(self.layer) or i >= len(self.layer[l]):
            return -1
        prev = l -1
        total = 0 
        for k in xrange(len(self.layer[prev])):
            a = self.active(prev,k)
            #sys.stderr.write("%d,%d,%d\n"%(prev,k,i))
            w = self.get_w(prev,k,i)
            total += w*a
        self.layer[l][i].z = total
        return 0
    def foward(self, record):
        self.clear_input_layer()
        for slot in record.vec:
            if slot >= len(self.layer[0]) or slot <= 0:
                continue
            self.layer[0][slot].z = record.vec[slot]
        for l in xrange(1,len(self.layer)):
            begin = 1
            if self.is_output_layer(l):
                begin = 0
            for i in xrange(begin,len(self.layer[l])):
                self.forward_update_node(l,i)    

    def clear_input_layer(self):
        for i in xrange(1,len(self.layer[0])):
            self.layer[0][i].set_z(0)
            
    def cal_grad(self,l,i,j):
        m = self.active(l,i)
        v = self.get_sigma(l+1,j)*m
        self.update_delta(l,i,j,v)

    def backward_update_out(self,y):
        nt = len(self.layer) - 1
        out_num = len(self.layer[nt])
        for i in xrange(out_num):
            loss = self.active(nt,i) - y[i]
            #loss = (self.active(nt,i) - y[i]) * self.gradient(z)
            self.set_sigma(nt,i,loss)
        
    def backward_update_in_node(self,l,i):
        #update nodes in input or hidden layer
        #nl short for next layer
        nl = l + 1
        begin = 1
        total = 0
        end = len(self.layer[nl])
        if self.is_output_layer(nl):
            begin = 0
        for j in xrange(begin,end):
            #compute the gradint of w(l,i,j)
            #cal_grad(l,i,j)
            w = self.get_w(l,i,j)
            s = self.get_sigma(nl,j)
            total += w * s
        loss = total * self.gradient(l,i)
        self.set_sigma(l,i,loss)

    def backward(self,y):
        self.backward_update_out(y)
        layer_num = len(self.layer)
        for l in xrange(layer_num-2, 0, -1):
            for i in xrange(1,len(self.layer[l])):
                self.backward_update_in_node(l,i)
        #Visit whole network, update edge delta gradient
        for l in xrange(layer_num-1):
            begin = 1
            if self.is_output_layer(l+1):
                begin = 0
            for i in xrange(len(self.layer[l])):
                for j in xrange(begin, len(self.layer[l+1])):
                    self.cal_grad(l,i,j)
        
    def update_weight(self, batch_size):
        for l in xrange(len(self.layer) - 1):
            for i in xrange(len(self.layer[l])):
                for j in xrange(len(self.layer[l+1])):
                    w = self.get_w(l,i,j)
                    w = w - self.alpha * (self.get_delta(l,i,j) / batch_size + self.regularize("L2",l,i,j))
                    self.set_w(l,i,j,w)
                    self.set_delta(l,i,j,0.0)
    def batch_train(self, sample_list):
        los = 0
        for sa in sample_list:
           self.foward(sa)
           los += self.loss(sa.label)
           #self.print_nn()
           self.backward(sa.label)   
        self.update_weight(len(sample_list))
        #sys.stderr.write("loss:%f\n"%(los/len(sample_list)))
        #self.print_nn_weight()
    def print_nn(self):
        for l in xrange(len(self.layer)):
            for i in xrange(len(self.layer[l])):
                sys.stderr.write("layer=%d,node=%d,w_len=%d,z=%f\n"%(l,i,len(self.layer[l][i].w),self.get_z(l,i)))
    def print_nn_weight(self):
        for l in xrange(len(self.layer)-1):
            begin =1
            if self.is_output_layer(l+1):
                begin=0
            for i in xrange(len(self.layer[l])):
                for j in xrange(begin,len(self.layer[l+1])):
                    w = self.get_w(l,i,j)
                    sys.stderr.write("w[%d,%d,%d]=%f\n"%(l,i,j,w))
    def loss(self,y):
        out_l = len(self.layer) - 1
        los = 0
        for i in xrange(len(y)):
            t = y[i] - self.layer[out_l][i].get_z()    
            los += t*t
        return los/2
    def reg_loss(self,regstr):
        loss = 0.0
        for l in xrange(len(self.layer)-1):
            for i in xrange(len(self.layer[l])):
                for j in xrange(len(self.layer[l+1])):
                    w = self.get_w(l,i,j)
                    if regstr=="L2":
                        loss += w*w/2.0
                    elif regstr=="L1":
                        loss += math.fabs(w)
        return loss*self.lam
    def log_loss(self,y):
        out_l = len(self.layer) - 1
        los = 0
        for i in xrange(len(y)):
            a = self.active(out_l,i)
            t = -(y[i] * math.log(a) +(1-y[i])* math.log(1-a))
            los+=t
        return los    
    def regularize(self,tp,l,i,j):
        if self.is_bias_node(l,i):
            return 0
        w = self.get_w(l,i,j)
        reg = 0
        if tp == "L2":
            reg = w
        elif tp == "L1":
            if w > 0:
                reg = 1
            elif w < 0:
                reg = -1
        return reg * self.lam

def grad_check(tnn, sa):
        tnn.clear_input_layer()
        tnn.clear_nn_delta()
        tnn.foward(sa)
        sys.stderr.write("%f\n"%(tnn.active(len(tnn.layer)-1,0)))
        tnn.backward(sa.label)
        epsion=0.0001
        for l in xrange(len(tnn.layer) - 1):
            begin = 1
            if tnn.is_output_layer(l+1):
                begin = 0
            for i in xrange(len(tnn.layer[l])):                
                for j in xrange(begin, len(tnn.layer[l+1])):
                    w_ori=tnn.get_w(l,i,j)
                    w=w_ori+epsion
                    tnn.set_w(l,i,j,w)
                    tnn.foward(sa)
                    los_pos=tnn.log_loss(sa.label)
                    w=w_ori-epsion
                    tnn.set_w(l,i,j,w)
                    tnn.foward(sa)
                    los_neg=tnn.log_loss(sa.label)
                    ng = float(los_pos-los_neg)/(epsion*2)
                    tnn.set_w(l,i,j,w_ori)
                    sys.stderr.write("w(%d,%d,%d):numer_grad=%f,real_grad=%f\n"%(l,i,j,ng,tnn.get_delta(l,i,j)))
def loss_all(nn,sample_all):
    los = 0
    for sa in sample_all:
        nn.foward(sa)
        los += nn.log_loss(sa.label)
    return los/len(sample_all) + nn.reg_loss("L2")

def train_global_loss(ff,nn):
    minibatch_size = 30
    sample_all=[]
    f = file(ff)
    for line in f:
        sa = gen_sample(line.strip())
        sample_all.append(sa)
    f.close()
    loss = loss_all(nn,sample_all)
    sys.stderr.write("original global loss:%.12f\n"%(loss))
    ind = 0
    while ind < len(sample_all):
        sample_list = sample_all[ind:ind+minibatch_size]
        nn.batch_train(sample_list)
        loss = loss_all(nn,sample_all)
        sys.stderr.write("global loss:%.12f\n"%(loss))
        ind += minibatch_size
def train_global_loss_2(ff,nn):
    minibatch_size = 10
    sample_all=[]
    f = file(ff)
    for line in f:
        sa = gen_sample(line.strip())
        sample_all.append(sa)
    f.close()
    loss = loss_all(nn,sample_all)
    sys.stderr.write("original global loss:%.12f\n"%(loss))
    ind = 0
    while ind < 100:
        nn.batch_train(sample_all)
        loss = loss_all(nn,sample_all)
        sys.stderr.write("global loss:%.12f\n"%(loss))
        ind += 1



 
def check(ff,nn):
    f = file(ff)
    for line in f:
        sa = gen_sample(line.strip())
        grad_check(nn,sa)
def train(ff,nn):
    minibatch_size = 10
    sample_list=[]
    f = file(ff)
    for line in f:
        if len(sample_list) == minibatch_size:
            nn.batch_train(sample_list)
            sample_list = []
        sa = gen_sample(line.strip())
        sample_list.append(sa)
    if len(sample_list) > 0:
            nn.batch_train(sample_list) 
    f.close()
    #nn.print_nn_weight()
def test(ff,nn):
    f = file(ff)
    for line in f:
        sa = gen_sample(line.strip())
        nn.foward(sa)
        predict = nn.active(len(nn.layer)-1,0)
        print "%.12f %s"%(predict,line.strip())
    f.close()

if __name__=="__main__":
    nn = NeuralNetwork()
    a = [4,3,1]
    nn.make_nn(a)
    nn.print_nn()
    nn.print_nn_weight()
    #train_global_loss("train",nn)
    #nn.print_nn()
    #nn.print_nn_weight()
    #test("test",nn)
    check("test",nn)
