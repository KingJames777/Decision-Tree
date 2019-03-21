from sklearn.model_selection import train_test_split as tts
from linear_CART import *
from ID3 import SplitDatasets
from numpy import *

def R2_score(pred,y):
      u=((y - pred) ** 2).sum() 
      v=((y- y.mean(axis=0)) ** 2).sum()
      return 1-u/v

##注意这里X和y放在一起
def load_data(filename):
      n=len(open(filename).readline().split('\t'))
      data=[]
      for line in open(filename).readlines():
            lineArr=[]
            curLine=line.strip().split('\t')
            for i in range(n):
                  lineArr.append(float(curLine[i]))
            data.append(lineArr)
      return array(data)

##区域的平方误差
def error(data):
      return var(data[:,-1])*shape(data)[0]

##叶节点的值
def leaf(data):
      return mean(data[:,-1])

##数据集；误差函数类型；叶节点函数类型
def best_split(data,errType=error,leafType=leaf,para=(1,4)):
      threErr=para[0]; threNum=para[1]  ## 误差减小阈值和最小叶节点数据数量
      if len(unique(data[:,-1]))==1 or len(data)<threNum:  ##  y都相等或数据过少直接返回叶节点
            return None, leafType(data)
      currErr=errType(data)  ## 划分前误差
      m,n=shape(data)
      bestErr=inf; feature=inf; value=inf
      for i in range(n-1):
            for split_value in set(data[:,i]):
                  index1,index2=SplitDatasets(i,split_value,data)
                  if len(index1)<threNum or len(index2)<threNum:  ##数据过少
                        continue
                  tempErr=errType(data[index1])+errType(data[index2])
                  if tempErr<bestErr:
                        feature=i
                        value=split_value
                        bestErr=tempErr
      if currErr-bestErr<threErr or bestErr==inf:  ##误差减小过少
            return None, leafType(data)
      return feature, value

def createTree(data,errType=error,leafType=leaf,para=(20,30)):
      feature, value=best_split(data,errType,leafType,para)
      if feature==None:
            return value
      Tree={}
      Tree['featIndex']=feature
      Tree['value']=value
      left,right=SplitDatasets(feature,value,data)
      Tree['left']=createTree(data[left],errType,leafType,para)
      Tree['right']=createTree(data[right],errType,leafType,para)
      return Tree

def predict(Tree,test):
      tree=Tree
      while isTree(tree):  ##要学会.
            if test[tree['featIndex']]<=tree['value']:
                  tree=tree['left']
            else:
                  tree=tree['right']
      return tree

def linearPred(Tree,test):
      coef=predict(Tree,test)
      return coef[:-1].dot(test[:-1])+coef[-1]

def isTree(obj):
      return type(obj).__name__=='dict'

##不推荐剪枝，CV确定最优参数更好
def prune(Tree,D_test):
      if isTree(Tree['left']) or isTree(Tree['right']):
            index1,index2=SplitDatasets(Tree['featIndex'],Tree['value'],D_test)
      if isTree(Tree['left']):  ##左子树仍是树
            Tree['left']=prune(Tree['left'],D_test[index1])
      if isTree(Tree['right']):
            Tree['right']=prune(Tree['right'],D_test[index2])
      if not isTree(Tree['left']) and not isTree(Tree['right']):  ##左右子树均是具体值
            index1,index2=SplitDatasets(Tree['featIndex'],Tree['value'],D_test)
            curErr=sum(((D_test[index1])[:,-1]-Tree['left'])**2)+\
                    sum(((D_test[index2])[:,-1]-Tree['right'])**2)
            mergedValue=(Tree['left']+Tree['right'])/2
            mergedErr=sum((D_test[:,-1]-mergedValue)**2)
            if mergedErr<curErr:
                  print('merging...')
                  return mergedValue
            else:  ##未合并
                  return Tree
      else:  ##一个子树是树，另一个是具体值
            return Tree

def CV(D_train,D_test,errType,leafType,predType):
      besErr=inf
      paras={}
      for s in arange(10,100,5,dtype='int8'):
            for t in arange(10,50,5,dtype='int8'):
                  Tree=createTree(D_train,errType,leafType,para=(s,t))
                  tempErr=0
                  for k in range(len(D_test)):
                        tempErr+=(predType(Tree,D_test[k])-D_test[k][-1])**2
                  if tempErr<besErr:
                        print(tempErr)
                        paras['threErr']=s
                        paras['threNum']=t
                        paras['tree']=Tree
                        besErr=tempErr
##      print(paras['threErr'],paras['threNum'])
      return paras['tree']

##比较CV和剪枝的效果
def test1(D_train,D_test):
      Tree_CV=CV(D_train,D_test,error,leaf,predict)

      Tree=createTree(D_train)
      Tree_pruned=prune(Tree,D_test)

      pred_CV=[]; pred_pruned=[]
      for i in range(len(D_test)):
            pred_CV.append(predict(Tree_CV,D_test[i]))
            pred_pruned.append(predict(Tree_pruned,D_test[i]))

      print('CV:',R2_score(pred_CV,D_test[:,-1]))  ##略好于剪枝结果
      print('Pruned:',R2_score(pred_pruned,D_test[:,-1]))

##线性回归树
def test2(D_train,D_test):
      Tree=CV(D_train,D_test,linearError,linearLeaf,linearPred)
      pred=[]
      for i in range(len(D_test)):
            pred.append(linearPred(Tree,D_test[i]))
      print(around(pred[:20]),'\n',D_test[:20,-1])
       
if __name__ == '__main__': 
      data=load_data('abalone.txt')
      m,n=shape(data)
      index=random.choice(range(m),size=int(m/5),replace=False)
      D_train=data[list(set(range(m))-set(index))]
      D_test=data[index]

##      test1(D_train,D_test)
      test2(D_train,D_test)
      
      












