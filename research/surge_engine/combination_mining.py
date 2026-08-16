from pathlib import Path
import pandas as pd
import numpy as np
import itertools, json

DATA=Path('research/surge_engine/data')
OUT=Path('research/surge_engine/results')
OUT.mkdir(parents=True,exist_ok=True)

FACTOR_COLS=['r3','r5','r10','r20','ma20_gap','ma60_gap','ma20_slope','dist20_high','dist60_high','atr14_pct','vol_ratio','vol_accel','upvol_ratio','rsi14','dd20','accel','range_pct','close_to_high']

def build_features():
    parts=[]
    for f in sorted(DATA.glob('kline_*.parquet')):
        x=pd.read_parquet(f,columns=['code','date','open','high','low','close','volume','amount'])
        x['date']=pd.to_datetime(x['date']); x=x.sort_values(['code','date'])
        g=x.groupby('code',group_keys=False); c=x['close'];h=x['high'];l=x['low'];v=x['volume']
        x['r3']=g['close'].pct_change(3); x['r5']=g['close'].pct_change(5); x['r10']=g['close'].pct_change(10); x['r20']=g['close'].pct_change(20)
        ma20=g['close'].transform(lambda s:s.rolling(20).mean()); ma60=g['close'].transform(lambda s:s.rolling(60).mean())
        x['ma20_gap']=c/ma20-1; x['ma60_gap']=c/ma60-1
        x['ma20_slope']=g['close'].transform(lambda s:s.rolling(20).mean()).groupby(x['code']).pct_change(20)
        x['dist20_high']=c/g['high'].transform(lambda s:s.rolling(20).max())-1; x['dist60_high']=c/g['high'].transform(lambda s:s.rolling(60).max())-1
        prev=g['close'].shift(1); tr=pd.concat([(h-l).abs(),(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
        x['atr14_pct']=tr.groupby(x['code']).transform(lambda s:s.rolling(14).mean())/c
        v20=g['volume'].transform(lambda s:s.rolling(20).mean());v5=g['volume'].transform(lambda s:s.rolling(5).mean())
        x['vol_ratio']=v/v20;x['vol_accel']=v5/v20
        up=v.where(g['close'].diff()>0,0);down=v.where(g['close'].diff()<0,0)
        x['upvol_ratio']=(up.groupby(x['code']).transform(lambda s:s.rolling(5).sum())+1)/(down.groupby(x['code']).transform(lambda s:s.rolling(5).sum())+1)
        delta=g['close'].diff();gain=delta.clip(lower=0).groupby(x['code']).transform(lambda s:s.rolling(14).mean());loss=(-delta.clip(upper=0)).groupby(x['code']).transform(lambda s:s.rolling(14).mean())
        rs=gain/loss.replace(0,np.nan);x['rsi14']=100-100/(1+rs)
        x['dd20']=c/g['close'].transform(lambda s:s.rolling(20).max())-1;x['accel']=x['r5']-x['r20']/4;x['range_pct']=(h-l)/c;x['close_to_high']=c/h-1
        x['amount20']=g['amount'].transform(lambda s:s.rolling(20).mean())
        x=x[(x.date>='2022-01-04')&(x.date<='2026-03-19')]
        parts.append(x)
    return pd.concat(parts,ignore_index=True)

feat=build_features()
labels=pd.read_parquet(OUT/'labels_all.parquet',columns=['code','date','mfe20','mae20','t20','mfe3','mfe5','mfe8','fail5'])
labels['date']=pd.to_datetime(labels['date'])
df=feat.merge(labels,on=['code','date'],how='inner').replace([np.inf,-np.inf],np.nan)
df=df[(df.close>=3)&(df.amount20>=2e7)&(df.volume>0)].copy()
train=df[(df.date>='2022-01-04')&(df.date<='2024-12-31')].copy();valid=df[(df.date>='2025-01-01')&(df.date<='2026-03-19')].copy()

# Select the strongest factors using train-only quantile-bin score.
def best_bin(s,f):
    s=s.dropna(subset=[f]);q=pd.qcut(s[f],q=10,duplicates='drop');g=s.assign(bin=q).groupby('bin',observed=True)
    z=g.agg(n=('mfe20','size'),mfe3=('mfe3','mean'),mfe5=('mfe5','mean'),mfe8=('mfe8','mean'),fail5=('fail5','mean'),mfe=('mfe20','mean'),mae=('mae20','mean')).reset_index()
    base=s.mfe3.mean();z['obj']=100*(z.mfe3-base)+50*z.mfe5+30*z.mfe8+20*z.mfe-80*z.fail5
    z=z.sort_values('obj',ascending=False)
    return z.iloc[0],q.cat.categories

single=[]; bins={}
for f in FACTOR_COLS:
    try:
        b,cats=best_bin(train,f); single.append((f,float(b.obj)));bins[f]=b
    except Exception:pass
single=sorted(single,key=lambda x:x[1],reverse=True)
top=[f for f,_ in single[:8]]

# Candidate rule for a factor is one of its best 3 train bins, but use numeric edges so validation is clean.
conds={}
for f in top:
    s=train.dropna(subset=[f]); edges=np.unique(s[f].quantile(np.linspace(0,1,11)).to_numpy()); cuts=pd.cut(s[f],bins=edges,include_lowest=True,duplicates='drop')
    z=s.assign(bin=cuts).groupby('bin',observed=True).agg(n=('mfe20','size'),mfe3=('mfe3','mean'),mfe5=('mfe5','mean'),mfe8=('mfe8','mean'),fail5=('fail5','mean'),mfe=('mfe20','mean')).reset_index()
    base=s.mfe3.mean();z['obj']=100*(z.mfe3-base)+50*z.mfe5+30*z.mfe8+20*z.mfe-80*z.fail5;z=z.sort_values('obj',ascending=False).head(3)
    arr=[]
    for _,r in z.iterrows():
        interval=r['bin']; arr.append({'lo':float(interval.left),'hi':float(interval.right),'train_obj':float(r.obj),'train_mfe3':float(r.mfe3),'train_mfe5':float(r.mfe5),'train_mfe8':float(r.mfe8),'train_fail5':float(r.fail5)})
    conds[f]=arr

def apply_rule(s, rule):
    mask=np.ones(len(s),dtype=bool)
    for f,c in rule:
        mask &= s[f].between(c['lo'],c['hi'],inclusive='both').to_numpy()
    return s.loc[mask]

def metrics(s):
    if len(s)<500:return None
    return {'n':int(len(s)),'mfe3':float(s.mfe3.mean()),'mfe5':float(s.mfe5.mean()),'mfe8':float(s.mfe8.mean()),'avg_mfe':float(s.mfe20.mean()),'fail5':float(s.fail5.mean()),'t20':float(s.t20.mean()),'score':float(100*s.mfe3.mean()+50*s.mfe5.mean()+30*s.mfe8.mean()+20*s.mfe20.mean()-80*s.fail5.mean())}

cands=[]
# beam-like search: start with best single rule, extend up to 4 factors using train objective.
start=[]
for f in top:
    for c in conds[f]: start.append([(f,c)])
beam=[]
for r in start:
    m=metrics(apply_rule(train,r))
    if m:beam.append((m['score'],r,m))
beam=sorted(beam,key=lambda x:x[0],reverse=True)[:40]
allres=[]
for depth in range(1,5):
    new=[]
    for _,rule,_ in beam:
        used={f for f,_ in rule}
        for f in top:
            if f in used:continue
            for c in conds[f]:
                rr=rule+[(f,c)];tm=metrics(apply_rule(train,rr))
                if tm:new.append((tm['score'],rr,tm))
    pool=beam+new
    # keep train diversity and rule count reasonable
    pool=sorted(pool,key=lambda x:x[0],reverse=True)[:60]
    beam=pool
    allres.extend(pool)

# Evaluate all candidates on validation and keep Pareto-optimal: cannot improve one of (mfe3,mfe5,mfe8,avg_mfe,-fail5) without worsening another.
results=[]
for _,rule,tm in allres:
    vm=metrics(apply_rule(valid,rule))
    if not vm:continue
    if vm['n']<300:continue
    results.append({'rule':rule,'train':tm,'valid':vm})

def dominates(a,b):
    A=a['valid'];B=b['valid']
    valsA=[A['mfe3'],A['mfe5'],A['mfe8'],A['avg_mfe'],-A['fail5']];valsB=[B['mfe3'],B['mfe5'],B['mfe8'],B['avg_mfe'],-B['fail5']]
    return all(x>=y for x,y in zip(valsA,valsB)) and any(x>y for x,y in zip(valsA,valsB))
pareto=[]
for r in results:
    if not any(dominates(o,r) for o in results if o is not r):pareto.append(r)
pareto=sorted(pareto,key=lambda r:(r['valid']['mfe3'],r['valid']['mfe5'],r['valid']['mfe8'],-r['valid']['fail5']),reverse=True)

out={'top_factors':top,'single_rank':single,'pareto':pareto[:30]}
(OUT/'combination_mining_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
print(json.dumps({'top_factors':top,'n_candidates':len(results),'n_pareto':len(pareto),'pareto':pareto[:10]},ensure_ascii=False,indent=2,default=str))
