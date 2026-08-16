from pathlib import Path
import pandas as pd
import numpy as np
import json

DATA=Path('research/surge_engine/data')
OUT=Path('research/surge_engine/results')
OUT.mkdir(parents=True,exist_ok=True)

files=sorted(DATA.glob('kline_*.parquet'))
parts=[]
for f in files:
    x=pd.read_parquet(f, columns=['code','date','open','high','low','close','volume','amount','turn','pctChg'])
    x['date']=pd.to_datetime(x['date']); x=x.sort_values(['code','date'])
    g=x.groupby('code',group_keys=False)
    c=x['close']; h=x['high']; l=x['low']; v=x['volume'];
    x['r3']=g['close'].pct_change(3); x['r5']=g['close'].pct_change(5); x['r10']=g['close'].pct_change(10); x['r20']=g['close'].pct_change(20)
    ma20=g['close'].transform(lambda s:s.rolling(20).mean()); ma60=g['close'].transform(lambda s:s.rolling(60).mean())
    x['ma20_gap']=c/ma20-1; x['ma60_gap']=c/ma60-1
    x['ma20_slope']=g['close'].transform(lambda s:s.rolling(20).mean()).groupby(x['code']).pct_change(20)
    x['dist20_high']=c/g['high'].transform(lambda s:s.rolling(20).max())-1
    x['dist60_high']=c/g['high'].transform(lambda s:s.rolling(60).max())-1
    prev=g['close'].shift(1); tr=pd.concat([(h-l).abs(),(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    x['atr14_pct']=tr.groupby(x['code']).transform(lambda s:s.rolling(14).mean())/c
    v20=g['volume'].transform(lambda s:s.rolling(20).mean()); v5=g['volume'].transform(lambda s:s.rolling(5).mean())
    x['vol_ratio']=v/v20; x['vol_accel']=v5/v20
    up=v.where(g['close'].diff()>0,0); down=v.where(g['close'].diff()<0,0)
    x['upvol_ratio']=(up.groupby(x['code']).transform(lambda s:s.rolling(5).sum())+1)/(down.groupby(x['code']).transform(lambda s:s.rolling(5).sum())+1)
    delta=g['close'].diff(); gain=delta.clip(lower=0).groupby(x['code']).transform(lambda s:s.rolling(14).mean()); loss=(-delta.clip(upper=0)).groupby(x['code']).transform(lambda s:s.rolling(14).mean())
    rs=gain/loss.replace(0,np.nan); x['rsi14']=100-100/(1+rs)
    x['dd20']=c/g['close'].transform(lambda s:s.rolling(20).max())-1; x['accel']=x['r5']-x['r20']/4
    x['range_pct']=(h-l)/c; x['close_to_high']=c/h-1
    x['amount20']=g['amount'].transform(lambda s:s.rolling(20).mean())
    x=x[(x.date>='2022-01-04')&(x.date<='2026-03-19')]
    # load labels from separately-generated parquet later by merge
    parts.append(x)

feat=pd.concat(parts,ignore_index=True)
labels=pd.read_parquet(OUT/'labels_all.parquet', columns=['code','date','mfe20','mae20','t20','mfe3','mfe5','mfe8','fail5'])
labels['date']=pd.to_datetime(labels['date'])
df=feat.merge(labels,on=['code','date'],how='inner')
df=df.replace([np.inf,-np.inf],np.nan)
# Baseline liquid universe: price >=3; nonzero turnover/amount; enough history implied by feature availability
# Keep two universes to detect whether liquidity filtering materially changes base rates.
df['liquid']=((df['close']>=3)&(df['amount20']>=2e7)&(df['volume']>0)).astype('int8')

def analyze(s, factor):
    s=s.dropna(subset=[factor,'mfe20','mae20']).copy()
    if len(s)<10000:return None
    # 10 equal-frequency bins; fit thresholds on train only
    s['bin']=pd.qcut(s[factor],q=10,duplicates='drop')
    g=s.groupby('bin',observed=True)
    z=g.agg(n=('mfe20','size'),mfe3=('mfe3','mean'),mfe5=('mfe5','mean'),mfe8=('mfe8','mean'),fail5=('fail5','mean'),avg_mfe=('mfe20','mean'),avg_mae=('mae20','mean'),t20=('t20','mean')).reset_index()
    base=float(s.mfe3.mean())
    z['lift']=z.mfe3-base
    z['score']=100*z.lift-35*z.fail5+8*z.t20.clip(lower=0)
    z=z.sort_values('score',ascending=False)
    best=z.iloc[0]
    # monotonicity via Spearman correlation between factor ranks and outcome
    corr=float(s[factor].rank(pct=True).corr(s['mfe3'],method='spearman'))
    return {'factor':factor,'n':len(s),'base_mfe3':base,'spearman_mfe3':corr,'best_bin':str(best['bin']),'best_n':int(best.n),'best_mfe3':float(best.mfe3),'best_mfe5':float(best.mfe5),'best_mfe8':float(best.mfe8),'best_fail5':float(best.fail5),'best_avg_mfe':float(best.avg_mfe),'best_avg_mae':float(best.avg_mae),'best_t20':float(best.t20),'lift':float(best.lift),'score':float(best.score),'bins':z.to_dict(orient='records')}

factors=['r3','r5','r10','r20','ma20_gap','ma60_gap','ma20_slope','dist20_high','dist60_high','atr14_pct','vol_ratio','vol_accel','upvol_ratio','rsi14','dd20','accel','range_pct','close_to_high']
# Train/validation split by date; thresholds are discovered in train only, then re-evaluated in validation.
train=df[(df.date>='2022-01-04')&(df.date<='2024-12-31')].copy()
valid=df[(df.date>='2025-01-01')&(df.date<='2026-03-19')].copy()
# use liquid train as primary study universe
train_liq=train[train.liquid==1].copy(); valid_liq=valid[valid.liquid==1].copy()
results=[]
for f in factors:
    r=analyze(train_liq,f)
    if r:
        # validation: compare same quantile bin based on train cutpoints
        sv=train_liq.dropna(subset=[f]); vv=valid_liq.dropna(subset=[f])
        edges=np.unique(sv[f].quantile(np.linspace(0,1,11)).to_numpy())
        if len(edges)>=3:
            cut=pd.cut(vv[f],bins=edges,include_lowest=True,duplicates='drop')
            vg=vv.assign(bin=cut).groupby('bin',observed=True).agg(n=('mfe20','size'),mfe3=('mfe3','mean'),mfe5=('mfe5','mean'),mfe8=('mfe8','mean'),fail5=('fail5','mean'),t20=('t20','mean')).reset_index()
            base=float(vv.mfe3.mean())
            vg['lift']=vg.mfe3-base; vg['score']=100*vg.lift-35*vg.fail5+8*vg.t20.clip(lower=0)
            bestv=vg.sort_values('score',ascending=False).iloc[0]
            r.update({'valid_base_mfe3':base,'valid_best_bin':str(bestv['bin']),'valid_best_n':int(bestv.n),'valid_best_mfe3':float(bestv.mfe3),'valid_best_mfe5':float(bestv.mfe5),'valid_best_mfe8':float(bestv.mfe8),'valid_best_fail5':float(bestv.fail5),'valid_best_t20':float(bestv.t20),'valid_lift':float(bestv.lift),'valid_score':float(bestv.score)})
        results.append(r)

results=sorted(results,key=lambda z:(z.get('valid_score',-999),z['score']),reverse=True)
summary={'train_rows':len(train_liq),'valid_rows':len(valid_liq),'train_base_mfe3':float(train_liq.mfe3.mean()),'valid_base_mfe3':float(valid_liq.mfe3.mean()),'ranking':[{k:r.get(k) for k in ['factor','score','lift','best_mfe3','best_mfe5','best_mfe8','best_fail5','best_t20','valid_score','valid_lift','valid_best_mfe3','valid_best_mfe5','valid_best_mfe8','valid_best_fail5','valid_best_t20','spearman_mfe3']} for r in results]}
(OUT/'factor_mining_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
for r in results:
    for b in r.get('bins',[]):
        b['factor']=r['factor']
    if r.get('bins'):
        pd.DataFrame(r['bins']).to_parquet(OUT/f"factor_{r['factor']}_bins.parquet",index=False)
print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
