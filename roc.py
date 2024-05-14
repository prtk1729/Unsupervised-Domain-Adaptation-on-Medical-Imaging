from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
def sig(x):
    return 1/(1 + np.exp(-x))

def probfun(a):
    return np.exp(a)/np.sum(np.exp(a))
def conv(x):
    if x==0:
        return 1
    elif x==1:
        return 0
lbl=[]
prob = []
#dataset_map/binary/revised_biobankjpg_valid_bin.txt
#dataset_map/binary/Foveal_valid_bin.txt
#dataset_map/binary/Biop_valid_bin.txt
with open('dataset_map/binary/Biop_valid_bin.txt') as f:
    for lb in f:   
        lbl.append(int(lb[-2]))
with open('rocprobs/probsFOBI.txt') as f1:
    for lb in f1:
        ar =np.array([float(lb.split(',')[0][1:]), float(lb.split(',')[1][:-1])])
        #print(ar)
        probret = sig(ar)
        #print(probret)
        prob.append(round(probret[1],3))
lbl=np.array(lbl)
prob = np.array(prob)
print(prob[:5])
print(lbl[:5])

fpr,tpr, th  = roc_curve(lbl, prob)
roc = roc_auc_score(lbl, prob)
print(f'roc_auc_score :{roc}')

prob=[]
with open('rocprobs/probsRBBI.txt') as f1:
    for lb in f1:
        ar =np.array([float(lb.split(',')[0][1:]), float(lb.split(',')[1][:-1])])
        #print(ar)
        probret = sig(ar)
        #print(probret)
        prob.append(round(probret[1],3))
lbl=np.array(lbl)
prob = np.array(prob)
print(prob[:5])

print(lbl[:5])
fpr1,tpr1, th1  = roc_curve(lbl, prob)
roc1=roc_auc_score(lbl, prob)
print(f'roc_auc_score1 :{roc1}')



plt.subplots(1, figsize=(12,12))
plt.title('Receiver Operating Characteristic FBI')
plt.plot(fpr, tpr,label="src:Foveal, AUC="+str(round(roc,3)))
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

#plt.plot(fpr1, tpr1,label="src:UKBiobank, AUC="+str(round(roc1,3)))
plt.legend()

plt.savefig('rocprobs/plots/ROC_fbi.png')
plt.show()
