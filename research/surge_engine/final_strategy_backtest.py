from pathlib import Path
import json, itertools
import numpy as np
import pandas as pd

DATA=Path('research/surge_engine/data'); OUT=Path('research/surge_engine/results'); OUT.mkdir(parents=True,exist_ok=True)
parts=[]
for f in sorted(DATA.glob('kline_*.parquet')):
    x=pd.read_parquet(f,columns=['code','date','open','high','low','close','volume','amount','pctChg'])
    x['date']=pd.to_datetime(x['date']); x=x.sort_values(['code','date'])
    g=x.groupby('code',group_keys=False); ma20=g.close.transform(lambda s:s.rolling(20).mean()); ma60=g.close.transform(lambda s:s.rolling(60).mean())
    x['r3']=g.close.pct_change(3); x['r5']=g.close.pct_change(5); x['r10']=g.close.pct_change(10); x['r20']=g.close.pct_change(20); x['ma20_gap']=x.close/ma20-1; x['ma60_gap']=x.close/ma60-1; x['ma20_slope']=ma20.groupby(x.code).pct_change(20); x['dist20_high']=x.close/g.close.transform(lambda s:s.rolling(20).max())-1; x['dist60_high']=x.close/g.close.transform(lambda s:s.rolling(60).max())-1; roll20=g.close.transform(lambda s:s.rolling(20).max()); x['dd20']=x.close/roll20-1; x['atr14_pct']=(g.high.transform(lambda s:s.rolling(14).max())-g.low.transform(lambda s:s.rolling(14).min()))/x.close; x['range_pct']=(x.high-x.low)/x.close; x['close_to_high']=x.close/x.high-1; x['vol_ratio']=x.volume/g.volume.transform(lambda s:s.rolling(20).mean()); x['upvol_ratio']=np.where(x.pctChg>0,x.volume,0)/g.volume.transform(lambda s:s.rolling(20).mean())
    x=x[(x.date>='2022-01-04')&(x.date<='2026-03-19')].copy(); parts.append(x)
df=pd.concat(parts,ignore_index=True).sort_values(['code','date']).reset_index(drop=True)
features=['r3','r5','r10','r20','ma20_gap','ma60_gap','ma20_slope','dist20_high','dist60_high','dd20','atr14_pct','range_pct','close_to_high','vol_ratio','upvol_ratio']; cut=pd.Timestamp('2025-01-01')
rows=[]
for code,gp in df.groupby('code',sort=False):
    gp=gp.reset_index(drop=True)
    for i in range(60,len(gp)-21):
        try: entry=float(gp.open.iloc[i+1]); hi=float(gp.high.iloc[i+1:i+21].max()); lo=float(gp.low.iloc[i+1:i+21].min())
        except (TypeError,ValueError): continue
        if not np.isfinite(entry) or entry<=0 or not np.isfinite(hi) or not np.isfinite(lo): continue
        feat=gp.iloc[i]; vals={}; ok=True
        for c in features:
            v=feat[c]
            if pd.isna(v): ok=False; break
            try: fv=float(v)
            except (TypeError,ValueError): ok=False; break
            if not np.isfinite(fv): ok=False; break
            vals[c]=fv
        if ok:
            future=gp.iloc[i+1:i+21]
            rows.append({'code':code,'date':feat.date,'entry':entry,**vals,'sample':'train' if feat.date<cut else 'valid','future_mfe':float(future.high.max()/entry-1),'future_mae':float(future.low.min()/entry-1)})
res=pd.DataFrame(rows); train=res[res.sample=='train'].copy(); valid=res[res.sample=='valid'].copy()
base_train=float(((train.future_mfe>=0.03)&(train.future_mae>-0.05)).mean()); base_valid=float(((valid.future_mfe>=0.03)&(valid.future_mae>-0.05)).mean())
single=[]
for f in features:
    qs=train[f].quantile(np.linspace(0,1,11)).drop_duplicates().to_numpy()
    if len(qs)<4: continue
    t=train.copy(); t['_bin']=pd.cut(t[f],bins=qs,include_lowest=True,duplicates='drop'); good=(t.future_mfe>=0.03)&(t.future_mae>-0.05); g=t.assign(good=good).groupby('_bin',observed=True).agg(n=('good','size'),good=('good','mean')).reset_index(); g['utility']=g.good-0.75/np.sqrt(g.n); best=g.sort_values(['utility','n'],ascending=[False,False]).iloc[0]; single.append((f,float(best['_bin'].left),float(best['_bin'].right),float(best.good),int(best.n)))
candidates=[]
for a,b in itertools.combinations(single,2):
    f1,lo1,hi1,_,_=a; f2,lo2,hi2,_,_=b; tr=train[train[f1].between(lo1,hi1)&train[f2].between(lo2,hi2)]
    if len(tr)<500: continue
    good=(tr.future_mfe>=0.03)&(tr.future_mae>-0.05); score=float(good.mean())+0.25*float((tr.future_mfe>=0.05).mean())+0.10*np.log1p(len(tr))/10; candidates.append({'factors':[f1,f2],'ranges':{f1:[lo1,hi1],f2:[lo2,hi2]},'train_n':len(tr),'train_good':float(good.mean()),'train_mfe3':float((tr.future_mfe>=0.03).mean()),'train_mfe5':float((tr.future_mfe>=0.05).mean()),'train_mfe8':float((tr.future_mfe>=0.08).mean()),'train_fail5':float((tr.future_mae<=-0.05).mean()),'score':score})
selected=sorted(candidates,key=lambda z:(z['score'],z['train_n']),reverse=True)[:10]

def simulate_rule(signals, trail_pct):
    trades=[]
    for code, ss in signals.groupby('code',sort=False):
        gp=df[df.code.eq(code)].reset_index(drop=True); idx={pd.Timestamp(d):i for i,d in enumerate(gp.date)}
        for _,s in ss.iterrows():
            i=idx.get(pd.Timestamp(s.date));
            if i is None or i+1>=len(gp): continue
            entry=float(gp.open.iloc[i+1]); stop=entry*0.95; mode='hard_stop'; peak=entry; exit_px=entry; reason='t20'; end=min(i+20,len(gp)-1)
            for j in range(i+1,end+1):
                hi=float(gp.high.iloc[j]); lo=float(gp.low.iloc[j]); cl=float(gp.close.iloc[j]); peak=max(peak,hi)
                if lo<=stop: exit_px=stop; reason=mode; break
                peak_mfe=peak/entry-1
                if peak_mfe>=0.05: stop=max(stop,peak*(1-trail_pct)); mode='trail'
                elif peak_mfe>=0.03: stop=max(stop,entry*1.01); mode='protect_1pct'
                exit_px=cl
            trades.append({'code':code,'signal_date':s.date,'entry':entry,'exit':exit_px,'ret':exit_px/entry-1,'reason':reason})
    return pd.DataFrame(trades)

results=[]
for rank,c in enumerate(selected,1):
    f1,f2=c['factors']; lo1,hi1=c['ranges'][f1]; lo2,hi2=c['ranges'][f2]; train_sig=train[train[f1].between(lo1,hi1)&train[f2].between(lo2,hi2)]; valid_sig=valid[valid[f1].between(lo1,hi1)&valid[f2].between(lo2,hi2)]
    for trail in [0.02,0.025,0.03]:
        for sample_name,sig in [('train',train_sig),('valid',valid_sig)]:
            tr=simulate_rule(sig,trail)
            if tr.empty: continue
            years=max(1e-9,(pd.to_datetime(sig.date.max())-pd.to_datetime(sig.date.min())).days/365.25) if len(sig) else 1.0
            results.append({'rank':rank,'factors':[f1,f2],'ranges':c['ranges'],'trail_pct':trail,'sample':sample_name,'n':len(tr),'annual_trades_est':float(len(tr)/years),'win_rate':float((tr.ret>0).mean()),'avg_ret':float(tr.ret.mean()),'median_ret':float(tr.ret.median()),'cum_compounded':float((1+tr.ret).prod()-1),'stop5_rate':float((tr.reason=='hard_stop').mean()),'protect_rate':float((tr.reason=='protect_1pct').mean()),'trail_rate':float((tr.reason=='trail').mean()),'t20_rate':float((tr.reason=='t20').mean())})
out={'objective':'Final OOS strategy backtest. Screening fitted on 2022-2024 only; T+1 open entry; initial -5% stop; after +3% set +1% protection; after +5% trail configured percentage; conservative same-day stop precedence. Nullable values skipped safely.','base_train_good':base_train,'base_valid_good':base_valid,'selected_train_rules':selected,'backtest':results}; (OUT/'final_strategy_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2))
