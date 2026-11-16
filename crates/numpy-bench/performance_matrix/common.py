"""Shared deterministic fixtures and untimed output fingerprints."""
from __future__ import annotations
import math
import numpy as np

def spec(group, feature, n, shape, dtype):
    return dict(id=f"{group}/{feature}/{n}",group=group,feature=feature,n=n,shape=list(shape),dtype=dtype,layout="C",numpy_api=f"np.{feature}",notes="",equivalence="equivalent")
def case(s, fn, snapshot=None):
    if snapshot is None: snapshot=snap
    result=fn(); check=snapshot(result); del result
    return dict(spec=s,fn=fn,check=check)
def array(n):
    return ((np.arange(n,dtype=np.int64)*37%1009).astype(np.float64)-504.)/128.
def positive(n):
    return 1.+(np.arange(n,dtype=np.int64)*37%1009).astype(np.float64)/1024.
def matrix(r,c): return array(r*c).reshape(r,c)
def scalar(x):
    x=float(x)
    if math.isnan(x): return "NaN"
    if math.isinf(x): return "Infinity" if x>0 else "-Infinity"
    return x
def snap(a):
    a=np.asarray(a); flat=a.ravel(order="C"); z=flat.astype(np.complex128)
    re=z.real;im=z.imag;finite=np.isfinite(re)&np.isfinite(im)
    n=a.size;indices=sorted(set([i*n//7 for i in range(7)]+[n-1])) if n else []
    # Fingerprints are computed outside timing. Match Rust's sequential sum to
    # avoid introducing a pairwise checksum-algorithm comparison.
    return dict(shape=list(a.shape),dtype=a.dtype.name,len=int(n),sum_re=scalar(sum(float(x) for x in re if np.isfinite(x))),sum_im=scalar(sum(float(x) for x in im if np.isfinite(x))),norm2=scalar(sum(float(x)*float(x)+float(y)*float(y) for x,y in zip(re[finite],im[finite]))),nan=int(np.count_nonzero(np.isnan(re)|np.isnan(im))),posinf=int(np.count_nonzero(np.isposinf(re)|np.isposinf(im))),neginf=int(np.count_nonzero(np.isneginf(re)|np.isneginf(im))),sample=[[scalar(re[i]),scalar(im[i])]for i in indices])
def snap_many(a): return [snap(x)for x in a]
