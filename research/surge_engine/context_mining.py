from pathlib import Path
import pandas as pd, numpy as np, itertools, json
DATA=Path('research/surge_engine/data'); OUT=Path('research/surge_engine/results'); OUT.mkdir(parents=True,exist_ok=True)

# Add market breadth + sector relative strength to the strongest price/volume factors.
parts=[]
for f in sorted(DATA.glob('kline_*.parquet')):
    x=pd.read_parquet(f,columns=['code','date','open','high','low','close','volume','amount'])
    x['date']=pd.to_datetime(x['date']); x=x.sort_values(['code','date'])
    g=x.groupby('code',group_keys=False); c=x.close; h=x.high; l=x.low; v=x.volume
    x['r5']=g.close.pct_change(5); x['r10']=g.close.pct_change(10); x['r20']=g.close.pct_change(20)
    ma20=g.close.transform(lambda s:s.rolling(20).mean()); ma60=g.close.transform(lambda s:s.rolling(60).mean())
    x['ma20_gap']=c/ma20-1; x['ma60_gap']=c/ma60-1
    x['ma20_slope']=g.close.transform(lambda s:s.rolling(20).mean()).groupby(x.code).pct_change(20)
    x['dist20_high']=c/g.high.transform(lambda s:s.rolling(20).max())-1; x['dist60_high']=c/g.high.transform(lambda s:s.rolling(60).max())-1
    prev=g.close.shift(1); tr=pd.concat([(h-l).abs(),(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    x['atr14_pct']=tr.groupby(x.code).transform(lambda s:s.rolling(14).mean())/c
    v20=g.volume.transform(lambda s:s.rolling(20).mean()); x['vol_ratio']=v/v20
    x['amount20']=g.amount.transform(lambda s:s.rolling(20).mean())
    delta=g.close.diff();gain=delta.clip(lower=0).groupby(x.code).transform(lambda s:s.rolling(14).mean());loss=(-delta.clip(upper=0)).groupby(x.code).transform(lambda s:s.rolling(14).mean())
    rs=gain/loss.replace(0,np.nan); x['rsi14']=100-100/(1+rs)
    x=x[(x.date>='2022-01-04')&(x.date<='2026-03-19')]
    parts.append(x)
feat=pd.concat(parts,ignore_index=True)

labels=pd.read_parquet(OUT/'labels_all.parquet',columns=['code','date','mfe20','mae20','t20','mfe3','mfe5','mfe8','fail5']); labels.date=pd.to_datetime(labels.date)
meta=pd.read_parquet(DATA/'stock_list.parquet')
meta['code']=meta['code'].astype(str).str.zfill(6)
# Sector column in public dataset is static; use only as an exploratory context factor.
df=feat.merge(labels,on=['code','date']).merge(meta[['code','sector']],on='code',how='left').replace([np.inf,-np.inf],np.nan)
df=df[(df.close>=3)&(df.amount20>=2e7)&(df.volume>0)].copy()

# Cross-sectional market environment, calculated only from same-day information.
df['above20']=(df.close > df.groupby('code').close.transform(lambda s:s.rolling(20).mean())).astype(int)
df['market_breadth20']=df.groupby('date').above20.transform('mean')
df['market_median_r5']=df.groupby('date').r5.transform('median')
df['market_median_r20']=df.groupby('date').r20.transform('median')
df['market_up_ratio']=df.groupby('date').r5.transform(lambda s:(s>0).mean())

# Sector median returns and stock-vs-sector relative strength.
sec= df.groupby(['date','sector'],dropna=False).close.median().reset_index().sort_values(['sector','date'])
sec['sec_r5']=sec.groupby('sector').close.pct_change(5); sec['sec_r20']=sec.groupby('sector').close.pct_change(20)
df=df.merge(sec[['date','sector','sec_r5','sec_r20']],on=['date','sector'],how='left')
df['sector_rel5']=df.r5-df.sec_r5; df['sector_rel20']=df.r20-df.sec_r20

# Two market regime factors as percentile ranks within date.
for col in ['market_breadth20','market_median_r5','market_median_r20','market_up_ratio','sector_rel5','sector_rel20']:
    df[col+'_pct']=df.groupby('date')[col].rank(pct=True)

train=df[(df.date>='2022-01-04')&(df.date<='2024-12-31')].copy(); valid=df[(df.date>='2025-01-01')&(df.date<='2026-03-19')].copy()
base_factor=['ma60_gap','ma20_slope','r20','dist60_high','ma20_gap','atr14_pct','vol_ratio','r5']
ctx=['market_breadth20','market_median_r5','market_median_r20','sector_rel5','sector_rel20','market_breadth20_pct','sector_rel5_pct','sector_rel20_pct']
factors=base_factor+ctx

def bins(s,f):
    s=s.dropna(subset=[f]);
    q=pd.qcut(s[f],q=8,duplicates='drop')
    return q.cat.categories

def metrics(s):
    if len(s)<300:return None
    return {'n':int(len(s)),'mfe3':float(s.mfe3.mean()),'mfe5':float(s.mfe5.mean()),'mfe8':float(s.mfe8.mean()),'avg_mfe':float(s.mfe20.mean()),'fail5':float(s.fail5.mean()),'t20':float(s.t20.mean()),'score':float(120*s.mfe3.mean()+55*s.mfe5.mean()+35*s.mfe8.mean()+20*s.mfe20.mean()-120*s.fail5.mean())}

def candidates(s,f):
    out=[]
    for iv in bins(s,f):
        z=s[s[f].between(iv.left,iv.right,inclusive='both')]; m=metrics(z)
        if m: out.append({'lo':float(iv.left),'hi':float(iv.right),'train':m})
    return sorted(out,key=lambda z:z['train']['score'],reverse=True)[:3]

cand={f:candidates(train,f) for f in factors}
# Seed with best single rules, beam-search to 4 factors.
beam=[]
for f,arr in cand.items():
    for c in arr: beam.append([(f,c)])
beam=sorted([(metrics(train) if False else c['train']['score'],r) for r in beam],key=lambda x:x[0],reverse=True)[:50]
allrules=[]
def apply(s,r):
    mask=np.ones(len(s),dtype=bool)
    for f,c in r: mask &= s[f].between(c['lo'],c['hi'],inclusive='both').to_numpy()
    return s.loc[mask]
for depth in range(1,5):
    new=[]
    for _,r in beam:
        used={f for f,_ in r}
        for f in factors:
            if f in used:continue
            for c in cand[f]:
                rr=r+[(f,c)]; tm=metrics(apply(train,rr))
                if tm and tm['n']>=500:new.append((tm['score'],rr,tm))
    pool=[]
    for _,r in beam:
        tm=metrics(apply(train,r));
        if tm:pool.append((tm['score'],r,tm))
    pool += new
    pool=sorted(pool,key=lambda x:x[0],reverse=True)[:80]
    beam=[(x[0],x[1]) for x in pool[:60]]
    allrules.extend(pool[:60])

res=[]
for _,r,tm in allrules:
    vm=metrics(apply(valid,r))
    if vm and vm['n']>=300:
        res.append({'rule':r,'train':tm,'valid':vm})

def dom(a,b):
    A=a['valid'];B=b['valid']; va=[A['mfe3'],A['mfe5'],A['mfe8'],A['avg_mfe'],-A['fail5']];vb=[B['mfe3'],B['mfe5'],B['mfe8'],B['avg_mfe'],-B['fail5']]
    return all(x>=y for x,y in zip(va,vb)) and any(x>y for x,y in zip(va,vb))
pareto=[r for r in res if not any(dom(o,r) for o in res if o is not r)]
pareto=sorted(pareto,key=lambda r:(r['valid']['mfe3'],r['valid']['mfe5'],r['valid']['mfe8'],-r['valid']['fail5']),reverse=True)
out={'factors':factors,'pareto':pareto[:40]}
(OUT/'context_mining_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
print(json.dumps({'n_candidates':len(res),'n_pareto':len(pareto),'pareto':pareto[:10]},ensure_ascii=False,indent=2,default=str))
