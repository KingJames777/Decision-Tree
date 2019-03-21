from numpy import *

def linearSolver(data):
      m=shape(data)[0]
      lamda=1e-12  ##防止不可求逆
      X=append(data[:,:-1],ones(m)[:,None],axis=1)
      y=data[:,-1]
      if linalg.det(X.T.dot(X))==0:
            coef=linalg.inv(X.T.dot(X)+lamda*eye(shape(X)[1])).dot(X.T).dot(y)
      else:
            coef=linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
      return X,y,coef

def linearLeaf(data):
      X,y,coef=linearSolver(data)
      return coef

def linearError(data):
      X,y,coef=linearSolver(data)
      return sum((X.dot(coef)-y)**2)
