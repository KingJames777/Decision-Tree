from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier as dtc
import numpy as np
import matplotlib.pyplot as plt

plotstep=0.02 #设定步进
plot_colors='ryb' #样本颜色

iris=load_iris()  ##这是个字典
##print(iris.data.shape)

for pairidx,pair in enumerate([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]):
  ##pairidx=index=0,1,2,3,4,5       pair=[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]
  X=iris.data[:,pair]
  y=iris.target

  clf=dtc()  ##可合并成clf=dtc().fit(X,y)
  clf.fit(X,y)
##  print(clf.predict([[5,5],[6,6]]))

  plt.subplot(2,3,pairidx+1)  ##第pairidx+1个子图
  xmin,xmax=X[:,0].min()-1,X[:,0].max()+1  ##设定x,y轴的区间
  ymin,ymax=X[:,1].min()-1,X[:,1].max()+1
  plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=1.5)  ##调节显示比例，参数为行距，列距，页边距
  xx,yy=np.meshgrid(np.arange(xmin,xmax,plotstep),np.arange(ymin,ymax,plotstep))
  zz=clf.predict(np.vstack((xx.ravel(),yy.ravel())).T).reshape(xx.shape)
  plt.contourf(xx,yy,zz,cmap=plt.cm.Spectral) #跟contour的区别在于它会填充色块
  plt.xlabel(iris.feature_names[pair[0]])
  plt.ylabel(iris.feature_names[pair[1]])

  for i,color in zip(range(3),plot_colors):
    ##zip将两个可迭代的对象逐对凑成元组，可通过list()转化成列表
    ##此处i,color就分别是(0,'r'),(1,'y'),(2,'b')
    index=np.where(y==i)  ##返回i类样本的索引
    plt.scatter(X[index,0],X[index,1],marker='x',s=15,edgecolor='k',c=color,cmap=plt.cm.RdYlBu)

plt.legend(loc='lower right')
plt.suptitle("Decision surface of a decision tree using paired features")  ##总标题,注意是sup...
plt.show()
