
import pandas as pd
import numpy as np
import gp_lib


import importlib
gp_lib = importlib.reload(gp_lib)


df = pd.read_csv('sotonmet.txt')


t_tmp = pd.to_numeric(pd.to_datetime(df['Reading Date and Time (ISO)']))/int(1e9)
t_tmp -= t_tmp[0]
y_tmp = df['Tide height (m)']
data = pd.concat([t_tmp, y_tmp], axis=1)


# get nan rows:
test = data[~pd.notna(y_tmp)].to_numpy()
train = data[pd.notna(y_tmp)].to_numpy()


X,y = train[:,0], train[:,1]
Xt = test[:,0]


scal = gp_lib.Scalar()
scal.fit(X,y)


X,y = scal.transform(X,y)
Xt, _ = scal.transform(Xt,y)

regr = gp_lib.GaussianProcess(
    kern=gp_lib.RbfKern(
        len=1,
        var=0.4,
        nvar=0.01,
    )
).fit(X,y)
yt, vart = regr.predict(Xt)
Xt,yt = scal.reverse(Xt,yt)
vart = scal.y_std**2*vart

