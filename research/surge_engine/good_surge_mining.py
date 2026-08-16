from pathlib import Path
import json, numpy as np, pandas as pd

DATA=Path('research/surge_engine/data'); OUT=Path('research/surge_engine/results'); OUT.mkdir(parents=True,exist_ok=True)
parts=[]
for f in sorted(DATA.glob('kline_*.parquet')):
    x=pd.read_parquet(f,columns=['code','date','open','high','low','close','volume','amount','pctChg'])
    x['date']=pd.to_datetime(x['date']); x=x.sort_values(['code','date'])
    g=x.groupby('code',group_keys=False)
    ma5=g.close.transform(lambda s:s.rolling(5).mean()); ma20=g.close.transform(lambda s:s.rolling(20).mean()); ma60=g.close.transform(lambda s:s.rolling(60).mean())
    x['r3']=g.close.pct_change(3); x['r5']=g.close.pct_change(5); x['r10']=g.close.pct_change(10); x['r20']=g.close.pct_change(20)
    x['ma20_gap']=x.close/ma20-1; x['ma60_gap']=x.close/ma60-1
    x['ma20_slope']=ma20.groupby(x.code).pct_change(20)
    x['dist20_high']=x.close/g.close.transform(lambda s:s.rolling(20).max())-1
    x['dist60_high']=x.close/g.close.transform(lambda s:s.rolling(60).max())-1
    roll20max=g.close.transform(lambda s:s.rolling(20).max()); x['dd20']=x.close/roll20max-1
    x['atr14_pct']=(g.high.transform(lambda s:s.rolling(14).max())-g.low.transform(lambda s:s.rolling(14).min()))/x.close
    x['range_pct']=(x.high-x.low)/x.close
    x['close_to_high']=x.close/x.high-1
    x['vol_ratio']=x.volume/g.volume.transform(lambda s:s.rolling(20).mean())
    x['upvol_ratio']=np.where(x.pctChg>0,x.volume,0)/g.volume.transform(lambda s:s.rolling(20).mean())
    x=x[(x.date>='2022-01-04')&(x.date<='2026-03-19')].copy(); parts.append(x)
df=pd.concat(parts,ignore_index=True).sort_values(['code','date']).reset_index(drop=True)

g=df.groupby('code',group_keys=False)
rows=[]
for code,gp in df.groupby('code',sort=False):
    gp=gp.reset_index(drop=True)
    for i in range(max(60,0),len(gp)-21):
        entry=float(gp.open.iloc[i+1]); hi=float(gp.high.iloc[i+1:i+21].max()); lo=float(gp.low.iloc[i+1:i+21].min())
        if not np.isfinite(entry) or entry<=0: continue
        mfe=hi/entry-1; mae=lo/entry-1
        feat=gp.iloc[i]
        rows.append({'code':code,'date':feat.date,'mfe':mfe,'mae':mae,'good_surge':int(mfe>=0.03 and mae>-0.05),'prime_surge':int(mfe>=0.05 and mae>-0.05),**{c:float(feat[c]) for c in ['r3','r5','r10','r20','ma20_gap','ma60_gap','ma20_slope','dist20_high','dist60_high','dd20','atr14_pct','range_pct','close_to_high','vol_ratio','upvol_ratio'] if np.isfinite(feat[c])}})
res=pd.DataFrame(rows)
features=['r3','r5','r10','r20','ma20_gap','ma60_gap','ma20_slope','dist20_high','dist60_high','dd20','atr14_pct','range_pct','close_to_high','vol_ratio','upvol_ratio']
cut=pd.Timestamp('2025-01-01'); res['sample']=np.where(res.date<cut,'train','valid')
base={p:{'n':int((res.sample==p).sum()),'good_rate':float(res.loc[res.sample==p,'good_surge'].mean()),'prime_rate':float(res.loc[res.sample==p,'prime_surge'].mean())} for p in ['train','valid']}
ranking=[]
for f in features:
    train=res[res.sample=='train'].dropna(subset=[f]); valid=res[res.sample=='valid'].dropna(subset=[f])
    bins=train[f].quantile(np.linspace(0,1,11)).drop_duplicates().to_numpy()
    if len(bins)<4: continue
    train=train.copy(); valid=valid.copy(); train['bin']=pd.cut(train[f],bins=bins,include_lowest=True,duplicates='drop'); valid['bin']=pd.cut(valid[f],bins=bins,include_lowest=True,duplicates='drop')
    tb=train.groupby('bin',observed=True).agg(n=('good_surge','size'),good=('good_surge','mean'),prime=('prime_surge','mean')).reset_index()
    best=tb.sort_values(['good','prime','n'],ascending=[False,False,False]).iloc[0]
    lo=float(best['bin'].left); hi=float(best['bin'].right); vv=valid[(valid[f]>=lo)&(valid[f]<=hi)]
    ranking.append({'factor':f,'train_good':float(best.good),'train_prime':float(best.prime),'train_n':int(best.n),'lo':lo,'hi':hi,'valid_good':float(vv.good_surge.mean()) if len(vv) else None,'valid_prime':float(vv.prime_surge.mean()) if len(vv) else None,'valid_n':int(len(vv))})
ranking=sorted(ranking,key=lambda z:(z['valid_good'] if z['valid_good'] is not None else -1,z['valid_prime'] if z['valid_prime'] is not None else -1,z['valid_n']),reverse=True)
# Simple train-fitted 2-factor screening from top 8 factors, using top training bin per factor; evaluate combinations on untouched validation.
top=[x['factor'] for x in ranking[:8]]; bestbins={x['factor']:(x['lo'],x['hi']) for x in ranking[:8]}
comb=[]
for i in range(len(top)):
    for j in range(i+1,len(top)):
        f1,f2=top[i],top[j]; a,b=bestbins[f1]; c,d=bestbins[f2]
        tr=res[(res.sample=='train')&(res[f1].between(a,b))&(res[f2].between(c,d))]; va=res[(res.sample=='valid')&(res[f1].between(a,b))&(res[f2].between(c,d))]
        if len(tr)<300 or len(va)<100: continue
        comb.append({'factors':[f1,f2],'train_n':len(tr),'train_good':float(tr.good_surge.mean()),'train_prime':float(tr.prime_surge.mean()),'valid_n':len(va),'valid_good':float(va.good_surge.mean()),'valid_prime':float(va.prime_surge.mean())})
comb=sorted(comb,key=lambda z:(z['valid_good'],z['valid_prime'],z['valid_n']),reverse=True)[:20]
out={'objective':'Predict Good Surge using T-close-only information: MFE>=3% AND MAE>-5%. No T+1 information is used as a predictor.','base':base,'single_factor_ranking':ranking,'top_factor_combinations':comb}
(OUT/'good_surge_mining_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2))
