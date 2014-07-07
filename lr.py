import math
import sys

g_alpha=0.01
g_lambda=1.0
g_fsize=0
g_w=[]
g_sample_list=[]
g_mx_round=100000

class Sample:
    def __init__(self):
        self.vec={}
        self.label=-1
    def vec_pos(self,x):
        if self.vec.has_key(x):
            return self.vec[x]
        else:
            return 0
def init():
    global g_w
    g_w=[0]*g_fsize

def gen_sample(line):
    sl=line.strip().split(" ")
    res=Sample()
    begin=0
    if sl[0].strip().find(":")<0:
        res.label=float(sl[0].strip())
        begin=1
    res.vec[0]=1
    for i in range(begin,len(sl)):
        ssl=sl[i].strip().split(":")
        try:
            k=int(ssl[0].strip())
            v=float(ssl[1].strip())
            res.vec[k] = v
        except:
            print line
    return res
def lin_theta(vec,theta):
    res=0.0
    for i in vec.keys():
        res += theta[i]*vec[i]
    return res
def sigmoid(x):
    return 1.0/(1.0+math.exp(-x))
def hypos(vec,theta):
    res=lin_theta(vec,theta)
    #print res
    return sigmoid(res)
def single_update(sam):
    h=hypos(sam.vec, g_w)     
    for i in sam.vec.keys():
        g_w[i] = g_w[i] - g_alpha * (h-sam.label) * sam.vec[i]
def single_update_L2(sam):
    h=hypos(sam.vec, g_w)     
    for i in sam.vec.keys():
        if i==0:
            g_w[i] = g_w[i] - g_alpha * (h-sam.label) * sam.vec[i]
        else:
            g_w[i] = g_w[i] - g_alpha * ((h-sam.label)*sam.vec[i] + g_lambda*g_w[i])
def batch_update(sam_list):
    sn=len(sam_list)
    if sn==0 :
        return
    wn=len(g_w)
    hy=[]
    for j in range(0,sn):
        h=hypos(sam_list[j].vec, g_w)
        hy.append(h)
    for i in range(0,wn):
        sum=0.0;
        for j in range(0,sn):
            if i in sam_list[j].vec:
                h=hy[j]
                sum += (h-sam_list[j].label) * sam_list[j].vec[i]
        
        g_w[i] = g_w[i] - g_alpha * sum / sn
def batch_update_L2(sam_list):
    sn=len(sam_list)
    if sn==0 :
        return
    wn=len(g_w)
    hy=[]
    for j in range(0,sn):
        h=hypos(sam_list[j].vec, g_w)
        hy.append(h)
    for i in range(0,wn):
        sum=0.0;
        for j in range(0,sn):
            if i in sam_list[j].vec:
                h=hy[j]
                sum += (h-sam_list[j].label) * sam_list[j].vec[i]
        if i==0:
            g_w[i] = g_w[i] - g_alpha * sum / sn
        else:
            g_w[i] = g_w[i] - g_alpha * (sum + g_lambda*g_w[i])/sn

def sgd_single(train_file):
    f = file(train_file) 
    for line in f:
        sam=gen_sample(line)
        single_update(sam)
    f.close();
def load_sample_list(train_file):
    f = file(train_file)
    for line in f:
        sam=gen_sample(line)
        g_sample_list.append(sam)
    f.close()
def loss():
    cost=0.0;
    if len(g_sample_list)==0:
        return 0;
    for i in g_sample_list:
        h=hypos(i.vec, g_w)
        c=0
        if i.label==1:
            c = 0-math.log(h)
        else:
            c = 0-math.log(1-h)
        cost+=c
    return cost/len(g_sample_list)
def loss_L2():
    cost=0.0;
    if len(g_sample_list)==0:
        return 0.0;
    for i in g_sample_list:
        h=hypos(i.vec, g_w)
        c=0
        if i.label==1:
            c = 0-math.log(h)
        else:
            c = 0-math.log(1-h)
        cost+=c
    wn=len(g_w)
    re=0.0
    for i in range(1,wn):
        re += g_w[i] * g_w[i]
    return (cost+g_lambda*re/2)/len(g_sample_list)
def det(sam_list):
    sn=len(sam_list)
    if sn==0:
        return
    wn=len(g_w)
    sum=0.0
    for i in range(0,sn):
        h=hypos(sam_list[i].vec, g_w)
        sum = sum + h - sam_list[i].label
    res=0.0
    for i in range(0,wn):
        r = (sum + g_lambda*g_w[i])/sn
        res += r*r
    return math.sqrt(res)
def over_cond2(round,sam_list):
    if round==0:
        return False 
    if round >= g_mx_round:
        return True
    d = det(sam_list)
    if d<0.001:
        return True
    return False
def over_cond(prev_loss,now_loss,round):
    if round==0:
        return False
    if now_loss>=0 and now_loss<=0.0001 :
        return True
    if prev_loss<=0:
        return False
    if round >= g_mx_round:
        return True
    ca=float(now_loss)-prev_loss
    if ca<0:
        return False
    r=math.fabs(ca)/prev_loss
    if r<0.001:
        return True
    return False

def sgd_menory(train_file):
    load_sample_list(train_file)
    sn=len(g_sample_list)
    if sn==0:
        return 
    prev_loss=-1
    now_loss=-1
    round=0

    #while not over_cond2(round,g_sample_list):
    while not over_cond(prev_loss,now_loss,round):
        prev_loss=now_loss
        j=round%sn
        sam=g_sample_list[j]
        single_update_L2(sam)
        now_loss=loss_L2()
        print(now_loss)
        round+=1
def bgd(train_file):
    load_sample_list(train_file)
    sn=len(g_sample_list)
    if sn==0:
        return 
    prev_loss=-1
    now_loss=-1
    round=0

    while not over_cond2(round,g_sample_list):
        #prev_loss=now_loss
        batch_update_L2(g_sample_list)
        now_loss=loss_L2()
        print(now_loss)
        round+=1

def train(ff):
    init()
    #print g_w
    #sgd_single(ff)
    sgd_menory(ff)
    #bgd(ff)
def save_model(ff):
    f=file(ff,"w")
    for i in g_w:
        f.write("%f\n"%(i))
    f.close()

def read_model(ff):
    global g_w
    global g_fsize
    f=file(ff)
    for line in f:
        g_w.append(float(line.strip()))
    g_fszie=len(g_w)
    f.close()
def predict(model,sam):
    r=hypos(sam.vec,model)
    sam.label=r
    return r 
if __name__=="__main__":
    if len(sys.argv)==1:
        print("please input train feature_size train_file model_file or predict model_file test_file out_file")
        sys.exit(-1)
    cmd=sys.argv[1]
    if cmd=="train":
        g_fsize=int(sys.argv[2])+1
        train_file=sys.argv[3]
        model_file=sys.argv[4]
        train(train_file)
        save_model(model_file)
    elif cmd=="predict":
        model_file=sys.argv[2]
        test_file=sys.argv[3]
        result_file=sys.argv[4]
        read_model(model_file)
        f=file(test_file)
        fo=file(result_file,"w")
        for line in f:
             sam=gen_sample(line)
             r=predict(g_w,sam)
             fo.write("%f %s\n"%(r,line.strip())) 
        f.close()
        fo.close()
    else:
        print("please input train or predict")
