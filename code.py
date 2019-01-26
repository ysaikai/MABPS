import os
import datetime
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import multiprocessing as mp


def func(x, a, b):
  xx = np.float64(-b*x) # This works to get around
  return a*(1-np.exp(xx))


def algorithm1(ii,iii):
  np.random.seed(ii) # seed = iter number (for reproducibility)
  T = TT[iii]
  arr_Z = np.array([0]*len(Z))

  rows_pool = np.array(range(num_sample))
  '''Test set'''
  rows_test = np.random.choice(rows_pool,num_test,replace=False)
  df_test = df_in.iloc[rows_test, :]
  df_test = df_test[["sr","yield"]]
  '''Validation set'''
  rows_pool = np.setdiff1d(rows_pool, rows_test)
  rows_valid = np.random.choice(rows_pool,num_valid,replace=False)
  df_valid = df_in.iloc[rows_valid, 0:2]

  '''Training set'''
  rows_train = np.setdiff1d(rows_pool, rows_valid)
  '''add two random samples to the selected set S'''
  S = pd.DataFrame(columns=df_test.columns) # empty
  samples = np.random.choice(rows_train, 2, replace=False)
  S = S.append(df_in.iloc[samples, 0:2])
  rows_train = np.setdiff1d(rows_train, samples)
  df_train = df_in.iloc[rows_train, :]
  '''for random selection'''
  S_rnd = S.copy() # with same two samples
  samples = np.random.choice(rows_train, T-2, replace=False)
  S_rnd = S_rnd.append(df_in.iloc[samples, 0:2])

  '''Segments'''
  locations = df_train.location.unique()
  pHs = df_train.pH.unique()
  A = [x for sublist in [locations, pHs] for x in sublist]
  K = len(A)
  d_df_seg = dict() # dict of dfs (one df for each segment)
  for location in locations:
    df = df_train.loc[df_train['location'] == location]
    d_df_seg[location] = df[["sr","yield"]]
  for pH in pHs:
    df = df_train.loc[df_train['pH'] == pH]
    d_df_seg[pH] = df[["sr","yield"]]

  a = dict(zip(A,[1]*K))
  b = dict(zip(A,[1]*K))
  q = dict(zip(A,[.5]*K)) # arbitrary
  SSE1 = np.inf # sum of squared error
  SSE2 = 0

  for t in range(2,T):
    for aa in A:
      q[aa] = np.random.beta(a[aa],b[aa])
    arm = max(q, key=q.get)
    arr_Z[Z.index(arm)] += 1 # for counting chosen arms
    df = d_df_seg[arm]
    indices = df.index.values
    ind = np.random.choice(indices)
    S = S.append(df.loc[ind])
    for aa in A: # remove the sample from all the segments
      df = d_df_seg[aa]
      d_df_seg[aa] = df[~df.index.isin([ind])]

    '''estimation'''
    xdata = np.array(S["sr"])
    ydata = np.array(S["yield"])
    (alpha,beta),pcov = curve_fit(func,xdata,ydata,bounds=(0,np.inf))

    '''reward'''
    xx = df_valid["sr"]
    yy = df_valid["yield"]
    SSE2 = sum(( yy-func(xx,alpha,beta) )**2)
    r = int(SSE2 < SSE1)
    if r == 1:
      a[arm] += 1
    else:
      b[arm] += 1
    SSE1 = SSE2 # for next loop

    '''remove empty segments'''
    AA = [_ for _ in A]
    for aa in AA:
      if d_df_seg[aa].empty:
        A.remove(aa)
        q[aa] = 0

  '''random selection'''
  xdata = np.array(S_rnd["sr"])
  ydata = np.array(S_rnd["yield"])
  popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, np.inf))
  alpha_rnd, beta_rnd = popt

  '''compare'''
  xx = df_test["sr"]
  yy = df_test["yield"]
  MSE_TS = ((yy-func(xx,alpha,beta))**2).mean()
  MSE_RND = ((yy-func(xx,alpha_rnd,beta_rnd))**2).mean()

  return (ii, iii, np.sqrt(MSE_TS), arr_Z, np.sqrt(MSE_RND))


'''
Main
'''
'''Parameters'''
TT = list(range(5,76,2)) # 5-75 w/ step 2
per_test = 0.3
per_valid = 0.2
iters = 10000
print("iters =", iters)
print("TT =", TT)


'''Initialization'''
os.chdir(os.path.dirname(os.path.abspath(__file__)))
df_in = pd.read_csv("input.csv")
num_sample = df_in.shape[0]
num_test = int(per_test*num_sample)
num_valid = int(per_valid*num_sample)

locations = df_in.location.unique()
pHs = df_in.pH.unique()
Z = [x for sublist in [locations, pHs] for x in sublist]
TS = np.zeros((iters,len(TT),1+len(Z))) # output for TS
RND = np.zeros((iters,len(TT))) # output for Random


'''Parallel processing'''
pool = mp.Pool(mp.cpu_count())
pairs = []
for iii in range(len(TT)):
  tmp = [(ii,iii) for ii in range(iters)]
  pairs += tmp
results = pool.starmap_async(algorithm1,pairs).get()
for r in results:
  TS[r[0],r[1],:] = np.concatenate([[r[2]],r[3]])
  RND[r[0],r[1]] = r[4]
pool.close()


'''Results'''
h = "t,RMSE,"
h = h + ','.join(map(str, Z))
tmp = np.column_stack([TT,TS.mean(axis=0)])
np.savetxt("TS.csv",tmp,fmt="%.4f",delimiter=",",header=h)
tmp = np.column_stack([TT,RND.mean(axis=0)])
np.savetxt("RND.csv",tmp,fmt="%.4f",delimiter=",",header=h)
