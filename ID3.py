import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score as acs
from sklearn.tree import DecisionTreeClassifier as dtc

##计算信息熵
def Entropy(y):
      keys,counts=np.unique(y,return_counts=True) ##y可取的值；每个值的次数
      N=counts.sum()
      classes=len(keys)
      ent=0
      for i in range(classes):
            prob=counts[i]/N
            ent+=prob*np.log2(prob)
      return -ent


##以某个属性的某个具体值划分数据集
def SplitDatasets(attribute_num,attribute_value,X):
      index1=np.where(X[:,attribute_num]<=attribute_value)[0]
      index2=np.where(X[:,attribute_num]>attribute_value)[0]
      return index1,index2


##计算某属性的所有划分点
def Split_criteria(attribute_num,X):
      res=[]
      temp=np.sort(X[:,attribute_num])
      for i in range(len(temp)-1):
            res.append((temp[i]+temp[i+1])/2)
      return res


##选择信息增益最大的属性
def Select_attribute(X,y):
      min_entropy=np.inf
      for attribute_num in range(X.shape[1]): ##每个属性
            split_criteria=Split_criteria(attribute_num,X)  ##划分点列表
            for attribute_value in split_criteria:  ##每个划分点
                  index1,index2=SplitDatasets(attribute_num,attribute_value,X)
                  y1,y2=y[index1],y[index2]
                  ent_sum=len(y1)/len(y)*Entropy(y1)+len(y2)/len(y)*Entropy(y2)
                  if ent_sum<min_entropy:
                        min_entropy=ent_sum
                        res_attribute_num=attribute_num
                        res_attribute_value=attribute_value
                        res_index1=index1
                        res_index2=index2 ##之前没写这两句 相当于直接返回最后的index难怪出错！
      entropy_gain=Entropy(y)-min_entropy
      return res_attribute_num,res_attribute_value,entropy_gain,res_index1,res_index2


##达不到划分阈值时给出的分类
def Default_class(y):
      keys,counts=np.unique(y,return_counts=True)  ##类别数；每类个数
      max_amount=counts.max() ##最多的类有多少个？
      for index in range(len(keys)):
            if counts[index]==max_amount:
                  return keys[index]


##构建决策树
def CreateDecisionTree(X,y,epsilon):
      y_list=list(y)
      if y_list.count(y_list[0])==len(y):  ##全是同类
            return y[0]
      attribute_num,attribute_value,entropy_gain,index1,index2=Select_attribute(X,y)
      if entropy_gain<epsilon:  ##小于划分阈值
            return Default_class(y)
      key=str(attribute_num)+' '+str(attribute_value)
      smaller,bigger='<=','>'
      Tree={key:{}}
      Tree[key][smaller]=CreateDecisionTree(X[index1],y[index1],epsilon)
      Tree[key][bigger]=CreateDecisionTree(X[index2],y[index2],epsilon)
      return Tree

##分类器
def Classifier(Tree,X):
      m,n=X.shape
      res=[]
      for index in range(m):
            tree=Tree
            while tree not in range(100):  ##是字典说明还未完成分类
                  comparison=next(iter(tree))  ##取出第一个判断条件
                  tree=tree[comparison]  ##第一个条件对应的字典
                  attribute_num=int(comparison.split()[0]) ##属性号
                  attribute_value=float(comparison.split()[1])  ##属性值
                  if X[index][attribute_num]<=attribute_value:
                        tree=tree['<=']
                  else:
                        tree=tree['>']
            res.append(tree)
      return res

if __name__ == '__main__':
      iris=datasets.load_wine() ##用于手写数字识别只有87%
      X=iris.data
      y=iris.target
      X_train,X_test,y_train,y_test=tts(X,y,test_size=0.2,random_state=19901120,stratify=y)
      
      epsilon=1e-2
      DT=CreateDecisionTree(X_train,y_train,epsilon)
      print(DT)
      print(acs(Classifier(DT,X_test),y_test))
      
      clf=dtc(criterion='entropy').fit(X_train,y_train)
      print(clf.score(X_test,y_test))















