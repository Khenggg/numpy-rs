"""Extended public API benchmark registry. Setup/checking are outside timing."""
import io
import os
from pathlib import Path
import tempfile

import numpy as np
from common import spec, case, snap, scalar, array, positive, matrix

LIB = '''sort argsort partition argpartition searchsorted lexsort top_k sort_complex sum prod min max ptp cumsum cumprod argmin argmax nanargmin nanargmax all any count_nonzero nansum nanprod nancumsum nancumprod nanmin nanmax mean nanmean var std nanvar nanstd quantile nanquantile percentile nanpercentile median nanmedian average cov corrcoef unique unique_all unique_counts unique_inverse unique_values isin union1d intersect1d setdiff1d setxor1d ediff1d histogram histogram_bin_edges histogram2d histogramdd bincount digitize convolve correlate interp diff gradient trapezoid unwrap polyval polyadd polysub polymul polydiv polyder polyint poly roots polyfit Polynomial Chebyshev Legendre Laguerre Hermite HermiteE polynomial_deriv polynomial_integ polynomial_roots series_to_polynomial poly1d write_npy read_npy save load savez savez_compressed loadz read_npz savetxt loadtxt dynamic_sum dynamic_mean dynamic_sort'''.split()
MASKED = 'filled compressed count add multiply sum prod min max mean var std map'.split()
STRINGS = 'upper lower strip lstrip rstrip replace split find startswith endswith zfill center ljust rjust join add'.split()
DATES = 'parse format cast add difference timedelta_cast timedelta_add is_busday busday_count busday_offset'.split()
RECORDS = 'zeros from_bytes field set_field'.split()
DIST = '''random uniform standard_normal normal standard_exponential exponential standard_gamma gamma beta binomial negative_binomial poisson geometric hypergeometric lognormal logseries chisquare noncentral_chisquare f noncentral_f standard_t standard_cauchy laplace logistic gumbel rayleigh weibull pareto power triangular vonmises wald zipf dirichlet multinomial multivariate_normal'''.split()
RANDOM_EXTRA = '''random_f32 standard_normal_f32 standard_exponential_f32 standard_gamma_f32 integers integers_inclusive integers_as_u32 bytes bytes_array random_scalar random_scalar_f32 standard_normal_scalar standard_normal_scalar_f32 standard_exponential_scalar standard_exponential_scalar_f32 choice_range choice_weighted choice_without_replacement choice permutation_range permutation permuted shuffle multinomial_array multivariate_normal_cholesky multivariate_normal_eigh multivariate_hypergeometric_marginals multivariate_hypergeometric_count normal_array uniform_array gamma_array beta_array binomial_array poisson_array triangular_array spawn par_random par_random_f32 par_standard_normal par_standard_normal_f32 par_standard_exponential par_standard_exponential_f32 par_normal par_uniform par_exponential par_gamma par_integers'''.split()
BITGEN = 'PCG64 PCG64DXSM MT19937 Philox SFC64 seed_state_u32 seed_state_u64 seed_spawn pcg_advance pcg_jumped pcgdxsm_advance pcgdxsm_jumped mt_jumped philox_advance philox_jumped'.split()
LEGACY_EXTRA = 'randint random_integers tomaxint bytes choice_range choice permutation_range permutation shuffle rand randn random_sample ranf sample'.split()
FFT = 'fft ifft fft2 ifft2 fftn ifftn rfft irfft rfft2 irfft2 rfftn irfftn hfft ihfft fftshift ifftshift fftfreq rfftfreq fftfreq_f32 rfftfreq_f32'.split()
BULK_LIB = set('sort sum mean var std cumsum nanmean quantile unique histogram convolve'.split())


def specs():
    out = []
    def add(group, feature, n, shape, dtype='float64'):
        s = spec(group, feature, n, shape, dtype)
        if group.startswith('extra.'):
            kind = group.split('.')[1]
            s['numpy_api'] = {'masked':'np.ma.MaskedArray.', 'strings':'np.strings.', 'datetime':'np.datetime64 / np.busday_', 'records':'np.ndarray structured dtype / '}[kind] + feature
        elif group.startswith('random'):
            s['numpy_api'] = ('np.random.RandomState(123).' if group == 'random.legacy' else 'np.random.Generator(np.random.PCG64(123)).') + feature
        elif group == 'fft':
            s['numpy_api'] = 'np.fft.' + feature.removesuffix('_f32')
            s['notes'] = 'Warm planner/cache; allocation included; norm=backward.'
        if feature.startswith('par_'):
            s['equivalence'] = 'distribution_only'
            s['notes'] = 'Rust spawned parallel streams versus NumPy serial Generator; checks only distribution invariants, not equal streams.'
        if group == 'extra.records' and feature == 'field':
            s['notes'] = 'Both return an independent copy of the numeric field; NumPy field view alone is not equivalent.'
        if group == 'extra.strings':
            s['notes'] = 'Variable-width UTF-8 StringDType; split uses object strings because np.strings.split has no StringDType loop.'
        out.append(s)
    for f in LIB:
        for n in ([1024,100000] if f in BULK_LIB else [1024]):
            add('lib',f,n,[8,n//8] if f in ['cov','corrcoef','histogramdd'] else [n])
    for g, features in [('extra.masked',MASKED),('extra.strings',STRINGS),('extra.datetime',DATES),('extra.records',RECORDS)]:
        for f in features:
            n = 1 if g == 'extra.datetime' else 1024
            add(g,f,n,[] if n==1 else [n], 'str' if g=='extra.strings' else ('datetime64[D]' if g=='extra.datetime' else 'float64'))
    for f in DIST + RANDOM_EXTRA:
        scalar_case = f.endswith('_scalar') or '_scalar_' in f
        for n in ([1] if scalar_case else ([1024,100000] if f in ['random','standard_normal','integers','gamma','par_random'] else [1024])):
            add('random',f,n,[] if scalar_case else ([n,3] if f in ['dirichlet','multinomial','multinomial_array','multivariate_hypergeometric_marginals','multivariate_hypergeometric_count'] else ([n,2] if f.startswith('multivariate_normal') else [n])), 'float32' if f.endswith('_f32') else ('int64' if f in ['integers','integers_inclusive','binomial','negative_binomial','poisson','geometric','hypergeometric','logseries','zipf'] else 'float64'))
    for f in BITGEN:
        add('random.bitgen',f,1024,[1024],'uint64')
    for f in DIST + LEGACY_EXTRA:
        add('random.legacy',f,1024,[1024])
    for f in FFT:
        for n in ([1024,131072] if f in ['fft','rfft'] else [1024]):
            add('fft',f,n,[8,8,16] if f.endswith('fftn') or f in ['fftn','ifftn','rfftn','irfftn'] else ([32,32] if f in ['fft2','ifft2','rfft2','irfft2'] else [n]), 'float32' if f.endswith('_f32') else 'float64')
    return out


def _string_snap(a):
    if isinstance(a, list):
        return {'len':len(a),'sample':a[:8]}
    a = np.asarray(a)
    return {'shape':list(a.shape),'len':a.size,'sample':a.ravel().tolist()[:8]}


def _masked_snap(a):
    return {'data':snap(np.ma.filled(a,0)), 'mask':snap(np.ma.getmaskarray(a))}


def _raw_snap(a):
    a = np.asarray(a).ravel()
    return {'len':a.size,'prefix':[int(x) for x in a[:16]]}


def _invariants(a):
    a=np.asarray(a)
    return {'shape':list(a.shape),'finite':bool(np.isfinite(a).all())}


def make(id):
    s = next((x for x in specs() if x['id']==id),None)
    if s is None:return None
    g,f,n=s['group'],s['feature'],s['n']
    if g == 'lib':return _lib(s,f,n)
    if g == 'fft':return _fft(s,f,n)
    if g == 'extra.masked':
        a=np.ma.array(array(n),mask=np.arange(n)%7==0);b=np.ma.array(positive(n),mask=np.arange(n)%11==0)
        if f=='map':return case(s,lambda:a*2,snapshot=_masked_snap)
        if f in ['filled','compressed','count']:return case(s,lambda:getattr(a,f)(**({'fill_value':-1.} if f=='filled' else {})),snapshot=lambda x:snap(np.asarray(x)))
        if f in ['add','multiply']:return case(s,lambda: a+b if f=='add' else a*b,snapshot=_masked_snap)
        return case(s,lambda:getattr(a,f)(),snapshot=_masked_snap)
    if g == 'extra.strings':
        a=np.array([f'  Ken {i%17} café  ' for i in range(n)],dtype=np.dtypes.StringDType());b=np.array('!',dtype=np.dtypes.StringDType())
        op={'upper':lambda:np.strings.upper(a),'lower':lambda:np.strings.lower(a),'strip':lambda:np.strings.strip(a),'lstrip':lambda:np.strings.lstrip(a),'rstrip':lambda:np.strings.rstrip(a),'replace':lambda:np.strings.replace(a,'Ken','Pet',count=1),'split':lambda:np.char.split(a.astype(object),None,1).tolist(),'find':lambda:np.strings.find(a,'café'),'startswith':lambda:np.strings.startswith(a,'  Ken'),'endswith':lambda:np.strings.endswith(a,'  '),'zfill':lambda:np.strings.zfill(a,24),'center':lambda:np.strings.center(a,24,fillchar='-'),'ljust':lambda:np.strings.ljust(a,24,fillchar='-'),'rjust':lambda:np.strings.rjust(a,24,fillchar='-'),'join':lambda:np.array(['-'.join(x) for x in a],dtype=np.dtypes.StringDType()),'add':lambda:np.strings.add(a,b)}
        if f in ['find','startswith','endswith']:return case(s,op[f])
        if f=='join':s['notes']+=' Python str.join per element: NumPy 2.5 has no native StringDType join ufunc.';s['equivalence']='python_fallback'
        return case(s,op[f],snapshot=_string_snap)
    if g == 'extra.datetime':
        date=np.datetime64('2024-03-29','D');other=np.datetime64('2024-04-08','D');delta=np.timedelta64(3,'h')
        op={'parse':lambda:np.datetime64('1969-12-31T23:59:59.123456789','ns').astype('i8'),'format':lambda:str(date),'cast':lambda:date.astype('datetime64[h]').astype('i8'),'add':lambda:(date+delta).astype('i8'),'difference':lambda:(other-date).astype('timedelta64[D]').astype('i8'),'timedelta_cast':lambda:np.timedelta64(-25,'h').astype('timedelta64[D]').astype('i8'),'timedelta_add':lambda:(np.timedelta64(1,'D')+delta).astype('i8'),'is_busday':lambda:np.is_busday(date),'busday_count':lambda:np.busday_count(date,other),'busday_offset':lambda:np.busday_offset(date,100,roll='forward').astype('i8')}
        return case(s,op[f],snapshot=(lambda x:str(x)) if f=='format' else (lambda x:scalar(float(x))))
    if g == 'extra.records':
        dtype=np.dtype([('tag','u1'),('value','<f8')]);a=np.zeros(n,dtype=dtype);a['value']=array(n);raw=a.tobytes();v=positive(n)
        if f=='zeros':return case(s,lambda:np.zeros(n,dtype=dtype),snapshot=lambda x:{'len':x.size,'itemsize':x.dtype.itemsize,'zero':not any(x.tobytes())})
        if f=='from_bytes':return case(s,lambda:np.frombuffer(raw,dtype=dtype).copy(),snapshot=lambda x:snap(x['value']))
        if f=='field':return case(s,lambda:a['value'].copy())
        def set_field():a['value']=v;return a
        return case(s,set_field,snapshot=lambda x:snap(x['value']))
    if g == 'random.bitgen':return _bitgen(s,f,n)
    if g in ['random','random.legacy']:return _random(s,f,n,g=='random.legacy')
    return None


def _lib(s,f,n):
    a=array(n);p=1.+(positive(n)-1.)/1000.;nan=a.copy();nan[::17]=np.nan;ints=(np.arange(n)*37%101).astype(np.int64)
    if f.startswith('dynamic_'):return case(s,lambda:getattr(np,f[8:])(a))
    if f in ['sort','argsort']:return case(s,lambda:getattr(np,f)(a,kind='stable'))
    if f=='partition':return case(s,lambda:np.partition(a,n//2),snapshot=lambda x:{'kvalue':float(x[n//2]),'partition_valid':bool((x[:n//2]<=x[n//2]).all() and (x[n//2:]>=x[n//2]).all()),'permutation_valid':bool(np.array_equal(np.sort(x),np.sort(a)))})
    if f=='argpartition':
        def check(x):
            v=a[x];return {'kvalue':float(v[n//2]),'partition_valid':bool((v[:n//2]<=v[n//2]).all() and (v[n//2:]>=v[n//2]).all()),'permutation_valid':bool(np.array_equal(np.sort(x),np.arange(n)))}
        return case(s,lambda:np.argpartition(a,n//2),snapshot=check)
    if f=='searchsorted':b=np.sort(a);return case(s,lambda:np.searchsorted(b,a,side='right'))
    if f=='lexsort':return case(s,lambda:np.lexsort((a,p)))
    if f=='top_k':return case(s,lambda:np.sort(np.partition(a,n-16)[-16:])[::-1])
    if f=='sort_complex':return case(s,lambda:np.sort_complex(a))
    if f in 'sum min max ptp cumsum argmin argmax all any count_nonzero mean var std median'.split():return case(s,lambda:getattr(np,f)(a),snapshot=lambda x:snap(np.asarray(x)))
    if f in ['prod','cumprod']:return case(s,lambda:getattr(np,f)(p))
    if f.startswith('nan') and f not in ['nanquantile','nanpercentile']:return case(s,lambda:getattr(np,f)(nan),snapshot=lambda x:snap(np.asarray(x)))
    if f in ['quantile','nanquantile','percentile','nanpercentile']:q=[10.,50.,90.] if 'percentile' in f else [.1,.5,.9];return case(s,lambda:getattr(np,f)(nan if f.startswith('nan') else a,q))
    if f=='average':w=positive(n);return case(s,lambda:np.average(a,weights=w),snapshot=lambda x:snap(np.asarray(x)))
    if f in ['cov','corrcoef']:m=matrix(8,n//8);return case(s,lambda:getattr(np,f)(m),snapshot=snap)
    if f in ['unique','unique_values']:return case(s,lambda:getattr(np,f)(ints))
    if f=='unique_all':return case(s,lambda:np.unique_all(ints),snapshot=lambda u:{'values':snap(u.values),'indices':snap(u.indices),'inverse':snap(u.inverse_indices),'counts':snap(u.counts)})
    if f in ['unique_counts','unique_inverse']:return case(s,lambda:getattr(np,f)(ints),snapshot=lambda u:{'values':snap(u.values),'other':snap(u[1])})
    if f=='isin':b=np.arange(0,101,3,dtype=np.int64);return case(s,lambda:np.isin(ints,b))
    if f in ['union1d','intersect1d','setdiff1d','setxor1d']:b=np.arange(50,150,dtype=np.int64);return case(s,lambda:getattr(np,f)(ints,b))
    if f=='ediff1d':return case(s,lambda:np.ediff1d(a))
    if f=='histogram':return case(s,lambda:np.histogram(a,bins=32,range=(-4.,4.)),snapshot=lambda x:{'counts':snap(x[0].astype(float)),'edges':snap(x[1])})
    if f=='histogram_bin_edges':return case(s,lambda:np.histogram_bin_edges(a,bins=32,range=(-4.,4.)))
    if f in ['histogram2d','histogramdd']:
        x=a[:n//2];y=a[n//2:]
        if f=='histogram2d':return case(s,lambda:np.histogram2d(x,y,bins=[16,16]),snapshot=lambda x:{'counts':snap(x[0]),'edges':[snap(x[1]),snap(x[2])]})
        sample=np.stack([x,y],axis=1);return case(s,lambda:np.histogramdd(sample,bins=[16,16]),snapshot=lambda x:{'counts':snap(x[0]),'edges':[snap(e) for e in x[1]]})
    if f=='bincount':return case(s,lambda:np.bincount(ints,minlength=128).astype(float))
    if f=='digitize':edges=np.linspace(-4.,4.,33);return case(s,lambda:np.digitize(a,edges))
    if f in ['convolve','correlate']:kernel=np.arange(17,dtype=float)/17.;return case(s,lambda:getattr(np,f)(a,kernel,mode='same'))
    if f=='interp':xp=np.linspace(-4.,4.,65);fp=xp*xp;return case(s,lambda:np.interp(a,xp,fp))
    if f=='diff':return case(s,lambda:np.diff(a,n=2))
    if f=='gradient':return case(s,lambda:np.gradient(a,0.5,edge_order=2))
    if f=='trapezoid':return case(s,lambda:np.trapezoid(a,dx=.5),snapshot=lambda x:snap(np.asarray(x)))
    if f=='unwrap':return case(s,lambda:np.unwrap(a))
    c=np.array([1.,-2.,.5,3.]);d=np.array([1.,.25,-.5])
    if f=='polyval':return case(s,lambda:np.polyval(c,a))
    if f in ['polyadd','polysub','polymul']:return case(s,lambda:getattr(np,f)(c,d))
    if f=='polydiv':return case(s,lambda:np.polydiv(c,d),snapshot=lambda x:[snap(x[0]),snap(x[1])])
    if f=='polyder':return case(s,lambda:np.polyder(c,2))
    if f=='polyint':return case(s,lambda:np.polyint(c,2,k=[0.]))
    if f=='poly':return case(s,lambda:np.poly([-2.,-1.,1.,2.]))
    if f=='roots':return case(s,lambda:np.sort_complex(np.roots([1.,0.,-1.])))
    if f=='polyfit':x=np.linspace(-1.,1.,n);y=2*x*x-3*x+1;return case(s,lambda:np.polyfit(x,y,2))
    if f in ['Polynomial','Chebyshev','Legendre','Laguerre','Hermite','HermiteE']:
        cls=getattr(np.polynomial,f);poly=cls(c);return case(s,lambda:poly(a))
    if f.startswith('polynomial_'):
        poly=np.polynomial.Polynomial(c)
        if f=='polynomial_deriv':return case(s,lambda:poly.deriv(2),snapshot=lambda p:snap(p.coef))
        if f=='polynomial_integ':return case(s,lambda:poly.integ(2),snapshot=lambda p:snap(p.coef))
        return case(s,lambda:np.sort_complex(poly.roots()))
    if f=='series_to_polynomial':poly=np.polynomial.Chebyshev(c);return case(s,lambda:poly.convert(kind=np.polynomial.Polynomial),snapshot=lambda p:snap(p.coef))
    if f=='poly1d':poly=np.poly1d(c);return case(s,lambda:poly(a))
    return _io(s,f,n,a)


def _io(s,f,n,a):
    path=Path(tempfile.gettempdir())/f'numpy-matrix-{os.getpid()}-{f}-{n}'
    s['notes']='Cached temporary-file I/O; fsync excluded. NPY memory cases include encoding/decoding, not setup.'
    if f in ['write_npy','read_npy']:
        buf=io.BytesIO();np.save(buf,a,allow_pickle=False);raw=buf.getvalue()
        if f=='read_npy':return case(s,lambda:np.load(io.BytesIO(raw),allow_pickle=False))
        def write():b=io.BytesIO();np.save(b,a,allow_pickle=False);return b.getvalue()
        return case(s,write,snapshot=lambda raw:snap(np.load(io.BytesIO(raw),allow_pickle=False)))
    if f in ['save','load']:path=path.with_suffix('.npy');np.save(path,a)
    elif f in ['savez','savez_compressed','loadz','read_npz']:path=path.with_suffix('.npz');np.savez(path,a=a)
    else:path=path.with_suffix('.txt');np.savetxt(path,a,fmt='%.18e',delimiter=',')
    if f=='load':return case(s,lambda:np.load(path,allow_pickle=False))
    if f=='save':return case(s,lambda:np.save(path,a,allow_pickle=False),snapshot=lambda _:snap(np.load(path,allow_pickle=False)))
    if f in ['savez','savez_compressed']:return case(s,lambda:getattr(np,f)(path,a=a),snapshot=lambda _: _npz_snap(path))
    if f=='loadz':return case(s,lambda:dict(np.load(path,allow_pickle=False)),snapshot=lambda a:{k:snap(v) for k,v in a.items()})
    if f=='read_npz':raw=path.read_bytes();return case(s,lambda:dict(np.load(io.BytesIO(raw),allow_pickle=False)),snapshot=lambda a:{k:snap(v) for k,v in a.items()})
    if f=='loadtxt':return case(s,lambda:np.loadtxt(path,delimiter=','))
    if f=='savetxt':return case(s,lambda:np.savetxt(path,a,fmt='%.18e',delimiter=','),snapshot=lambda _:snap(np.loadtxt(path,delimiter=',')))
    raise KeyError(f)


def _npz_snap(path):
    with np.load(path,allow_pickle=False) as z:return {k:snap(z[k]) for k in z.files}


def _fft(s,f,n):
    shape=s['shape'];a=array(n).reshape(shape);c=a.astype(complex)+.25j
    if f in ['fft','ifft']:return case(s,lambda:getattr(np.fft,f)(c))
    if f in ['fft2','ifft2','fftn','ifftn']:return case(s,lambda:getattr(np.fft,f)(c))
    if f in ['rfft','rfft2','rfftn','ihfft']:return case(s,lambda:getattr(np.fft,f)(a))
    if f in ['irfft','hfft']:spectrum=np.fft.rfft(a);return case(s,lambda:getattr(np.fft,f)(spectrum,n=n))
    if f=='irfft2':spectrum=np.fft.rfft2(a);return case(s,lambda:np.fft.irfft2(spectrum,s=shape))
    if f=='irfftn':spectrum=np.fft.rfftn(a);return case(s,lambda:np.fft.irfftn(spectrum,s=shape))
    if f in ['fftshift','ifftshift']:return case(s,lambda:getattr(np.fft,f)(a))
    if f.endswith('_f32'):
        s['notes']='NumPy fftfreq has no dtype argument; equivalent explicitly includes astype(float32).'
        return case(s,lambda:getattr(np.fft,f[:-4])(n,d=.25).astype(np.float32))
    return case(s,lambda:getattr(np.fft,f)(n,d=.25))


def _bitgen(s,f,n):
    if f in ['PCG64','PCG64DXSM','MT19937','Philox','SFC64']:
        bg=getattr(np.random,f)(123);return case(s,lambda:bg.random_raw(n),snapshot=_raw_snap)
    if f.startswith('seed_'):
        seed=np.random.SeedSequence(123)
        if f=='seed_spawn':return case(s,lambda:seed.spawn(4),snapshot=lambda children:[_raw_snap(c.generate_state(4)) for c in children])
        return case(s,lambda:seed.generate_state(n,dtype=np.uint32 if f.endswith('u32') else np.uint64),snapshot=_raw_snap)
    kind='PCG64DXSM' if f.startswith('pcgdxsm') else ('MT19937' if f.startswith('mt_') else ('Philox' if f.startswith('philox') else 'PCG64'))
    bg=getattr(np.random,kind)(123)
    if f.endswith('advance'):
        def advance():bg.advance(12345);return bg.random_raw(8)
        return case(s,advance,snapshot=_raw_snap)
    return case(s,lambda:bg.jumped(1).random_raw(8),snapshot=_raw_snap)


def _random(s,f,n,legacy=False):
    r=np.random.RandomState(123) if legacy else np.random.default_rng(123)
    shape=(n,);a=array(n);one=np.array(1.);two=np.array(2.);cov=np.array([[1.,.25],[.25,2.]])
    ops={'random':lambda:r.random(shape),'uniform':lambda:r.uniform(-2.,3.,shape),'standard_normal':lambda:r.standard_normal(shape),'normal':lambda:r.normal(1.5,2.,shape),'standard_exponential':lambda:r.standard_exponential(shape),'exponential':lambda:r.exponential(2.,shape),'standard_gamma':lambda:r.standard_gamma(2.,shape),'gamma':lambda:r.gamma(2.,3.,shape),'beta':lambda:r.beta(2.,5.,shape),'binomial':lambda:r.binomial(20,.3,shape),'negative_binomial':lambda:r.negative_binomial(4.,.4,shape),'poisson':lambda:r.poisson(8.,shape),'geometric':lambda:r.geometric(.3,shape),'hypergeometric':lambda:r.hypergeometric(20,30,10,shape),'lognormal':lambda:r.lognormal(.5,.75,shape),'logseries':lambda:r.logseries(.4,shape),'chisquare':lambda:r.chisquare(4.,shape),'noncentral_chisquare':lambda:r.noncentral_chisquare(4.,2.,shape),'f':lambda:r.f(5.,9.,shape),'noncentral_f':lambda:r.noncentral_f(5.,9.,2.,shape),'standard_t':lambda:r.standard_t(7.,shape),'standard_cauchy':lambda:r.standard_cauchy(shape),'laplace':lambda:r.laplace(1.,2.,shape),'logistic':lambda:r.logistic(1.,2.,shape),'gumbel':lambda:r.gumbel(1.,2.,shape),'rayleigh':lambda:r.rayleigh(2.,shape),'weibull':lambda:r.weibull(2.,shape),'pareto':lambda:r.pareto(3.,shape),'power':lambda:r.power(2.,shape),'triangular':lambda:r.triangular(-2.,.5,3.,shape),'vonmises':lambda:r.vonmises(.5,2.,shape),'wald':lambda:r.wald(2.,3.,shape),'zipf':lambda:r.zipf(2.,shape),'dirichlet':lambda:r.dirichlet([1.,2.,3.],shape),'multinomial':lambda:r.multinomial(20,[.2,.3,.5],shape),'multivariate_normal':lambda:r.multivariate_normal([1.,2.],cov,shape)}
    if f in ops:return case(s,ops[f])
    if f in ['bytes','bytes_array']:return case(s,lambda:np.frombuffer(r.bytes(n),dtype=np.uint8).copy(),snapshot=_raw_snap)
    if legacy:
        if f in ['randint','random_integers','tomaxint']:
            op=(lambda:r.randint(-20,50,shape,dtype=np.int64)) if f=='randint' else ((lambda:r.random_integers(-20,50,shape)) if f=='random_integers' else (lambda:r.tomaxint(shape)))
            return case(s,op,snapshot=_raw_snap if f=='tomaxint' else snap)
        if f in ['rand','randn']:return case(s,lambda:getattr(r,f)(n))
        if f in ['random_sample','ranf','sample']:return case(s,lambda:getattr(r,f)(shape))
    if f.endswith('_scalar') or '_scalar_' in f:
        name=f.replace('_scalar','');dtype=np.float32 if name.endswith('_f32') else None;name=name.removesuffix('_f32')
        return case(s,lambda:getattr(r,name)(dtype=dtype) if dtype else getattr(r,name)(),snapshot=lambda x:scalar(float(x)))
    if f in ['random_f32','standard_normal_f32','standard_exponential_f32','standard_gamma_f32']:
        name=f.removesuffix('_f32');return case(s,lambda:getattr(r,name)(2.,shape,dtype=np.float32) if name=='standard_gamma' else getattr(r,name)(shape,dtype=np.float32))
    if f in ['integers','integers_inclusive','integers_as_u32']:return case(s,lambda:r.integers(0,100,shape,dtype=np.uint32 if f=='integers_as_u32' else np.int64,endpoint=f=='integers_inclusive'))
    if f in ['choice_range','choice_weighted','choice_without_replacement','choice']:
        probs=np.arange(1,17,dtype=float);probs/=probs.sum()
        if f=='choice_weighted':return case(s,lambda:r.choice(16,shape,p=probs))
        if f=='choice_without_replacement':return case(s,lambda:r.choice(n,64,replace=False))
        return case(s,lambda:r.choice(a if f=='choice' else n,shape))
    if f in ['permutation_range','permutation','permuted']:return case(s,lambda:r.permuted(a) if f=='permuted' else r.permutation(n if f=='permutation_range' else a))
    if f=='shuffle':
        mutable=a.copy()
        def shuffle():r.shuffle(mutable);return None
        return case(s,shuffle,snapshot=lambda _:snap(mutable))
    if f=='multinomial_array':p=np.tile([.2,.3,.5],(n,1));return case(s,lambda:r.multinomial(20,p))
    if f.startswith('multivariate_normal_'):return case(s,lambda:r.multivariate_normal([1.,2.],cov,shape,method=f.rsplit('_',1)[1]))
    if f.startswith('multivariate_hypergeometric_'):return case(s,lambda:r.multivariate_hypergeometric([20,30,50],10,shape,method=f.rsplit('_',1)[1]))
    if f.endswith('_array'):
        names={'normal_array':lambda:r.normal(np.full(n,1.),np.full(n,2.)),'uniform_array':lambda:r.uniform(np.full(n,1.),np.full(n,2.)),'gamma_array':lambda:r.gamma(np.full(n,1.),np.full(n,2.)),'beta_array':lambda:r.beta(np.full(n,1.),np.full(n,2.)),'binomial_array':lambda:r.binomial(np.full(n,20,dtype=np.int64),np.full(n,.3)),'poisson_array':lambda:r.poisson(np.full(n,8.)),'triangular_array':lambda:r.triangular(np.full(n,-2.),np.full(n,.5),np.full(n,3.))}
        # Preallocate parameter arrays so setup is excluded on both sides.
        params={'normal_array':(np.full(n,1.),np.full(n,2.)),'uniform_array':(np.full(n,1.),np.full(n,2.)),'gamma_array':(np.full(n,1.),np.full(n,2.)),'beta_array':(np.full(n,1.),np.full(n,2.)),'binomial_array':(np.full(n,20,dtype=np.int64),np.full(n,.3)),'poisson_array':(np.full(n,8.),),'triangular_array':(np.full(n,-2.),np.full(n,.5),np.full(n,3.))}[f]
        return case(s,lambda:getattr(r,f[:-6])(*params))
    if f=='spawn':return case(s,lambda:r.spawn(4),snapshot=lambda children:[snap(c.random(8)) for c in children])
    if f.startswith('par_'):
        name=f[4:];name=name.removesuffix('_f32');dtype=np.float32 if f.endswith('_f32') else None
        par={'random':lambda:r.random(shape,dtype=dtype or np.float64),'standard_normal':lambda:r.standard_normal(shape,dtype=dtype or np.float64),'standard_exponential':lambda:r.standard_exponential(shape,dtype=dtype or np.float64),'normal':lambda:r.normal(1.5,2.,shape),'uniform':lambda:r.uniform(-2.,3.,shape),'exponential':lambda:r.exponential(2.,shape),'gamma':lambda:r.gamma(2.,3.,shape),'integers':lambda:r.integers(0,100,shape)}
        return case(s,par[name],snapshot=_invariants)
    raise KeyError(f)
