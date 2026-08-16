from pathlib import Path
import pandas as pd
import numpy as np
import json

DATA=Path('research/surge_engine/data')
OUT=Path('research/surge_engine/results')
OUT.mkdir(parents=True,exist_ok=True)

files=sorted(DATA.glob('kline_*.parquet'))
all_parts=[]
for f in files:
    x=pd.read_parquet(f, columns=['code','date','open','high','low','close','volume','amount','turn','pctChg'])
    x['date']=pd.to_datetime(x['date'])
    x=x.sort_values(['code','date'])
    all_parts.append(x)

df=pd.concat(all_parts,ignore_index=True)
df['code']=df['code'].astype(str).str.zfill(6)

# 以 T+1 开盘为入场点，未来20个交易日为标签窗口
parts=[]
for code,g in df.groupby('code',sort=False):
    g=g.sort_values('date').copy()
    n=len(g)
    ep=np.full(n,np.nan);mfe=np.full(n,np.nan);mae=np.full(n,np.nan);t20=np.full(n,np.nan)
    oh=g['open'].to_numpy(); hi=g['high'].to_numpy(); lo=g['low'].to_numpy(); cl=g['close'].to_numpy()
    for i in range(n-20):
        e=oh[i+1]
        if np.isfinite(e) and e>0:
            fut_hi=hi[i+1:i+21]; fut_lo=lo[i+1:i+21]; fut_cl=cl[i+1:i+21]
            if np.all(np.isfinite(fut_hi)) and np.all(np.isfinite(fut_lo)) and np.all(np.isfinite(fut_cl)):
                ep[i]=e; mfe[i]=np.max(fut_hi/e-1); mae[i]=np.min(fut_lo/e-1); t20[i]=fut_cl[-1]/e-1
    g['entry_open']=ep;g['mfe20']=mfe;g['mae20']=mae;g['t20']=t20
    g['mfe3']=(g['mfe20']>=0.03).astype('Int8')
    g['mfe5']=(g['mfe20']>=0.05).astype('Int8')
    g['mfe8']=(g['mfe20']>=0.08).astype('Int8')
    g['fail5']=(g['mae20']<=-0.05).astype('Int8')
    parts.append(g)

out=pd.concat(parts,ignore_index=True)
out=out[out['mfe20'].notna()].copy()
out.to_parquet(OUT/'labels_all.parquet',index=False)

# 研究窗口：2022-01-01 至 2026-03-31，避免最近20日标签不完整
r=out[(out.date>='2022-01-01')&(out.date<='2026-03-31')].copy()

summary={
 'rows':int(len(r)),
 'stocks':int(r.code.nunique()),
 'date_min':str(r.date.min().date()),
 'date_max':str(r.date.max().date()),
 'mfe3_rate':float(r.mfe3.mean()),
 'mfe5_rate':float(r.mfe5.mean()),
 'mfe8_rate':float(r.mfe8.mean()),
 'fail5_rate':float(r.fail5.mean()),
 'avg_mfe':float(r.mfe20.mean()),
 'median_mfe':float(r.mfe20.median()),
 'avg_mae':float(r.mae20.mean()),
 'median_mae':float(r.mae20.median()),
 'avg_t20':float(r.t20.mean()),
 'median_t20':float(r.t20.median()),
 'mfe3_avg_t20':float(r.loc[r.mfe3==1,'t20'].mean()),
 'mfe3_avg_mfe':float(r.loc[r.mfe3==1,'mfe20'].mean()),
 'mfe3_avg_mae':float(r.loc[r.mfe3==1,'mae20'].mean()),
 'lt3_avg_t20':float(r.loc[r.mfe3==0,'t20'].mean()),
 'lt3_avg_mfe':float(r.loc[r.mfe3==0,'mfe20'].mean()),
}
(OUT/'label_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
