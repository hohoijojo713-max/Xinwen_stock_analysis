from pathlib import Path
import pandas as pd, numpy as np, json

DATA=Path('research/surge_engine/data'); OUT=Path('research/surge_engine/results'); OUT.mkdir(parents=True,exist_ok=True)

# Daily-only executable entry research. Signal is formed at T close; execution is T+1.
# We explicitly mark a T+1 open pinned at the daily limit as NO_FILL.
parts=[]
for f in sorted(DATA.glob('kline_*.parquet')):
    x=pd.read_parquet(f,columns=['code','date','open','high','low','close','volume','amount','pctChg'])
    x['date']=pd.to_datetime(x['date'])
    x=x.sort_values(['code','date']).copy()
    g=x.groupby('code',group_keys=False)
    x['r5']=g.close.pct_change(5)
    x['r20']=g.close.pct_change(20)
    ma20=g.close.transform(lambda s:s.rolling(20).mean())
    ma60=g.close.transform(lambda s:s.rolling(60).mean())
    x['ma20_gap']=x.close/ma20-1
    x['ma60_gap']=x.close/ma60-1
    ma20_series=g.close.transform(lambda s:s.rolling(20).mean())
    x['ma20_slope']=ma20_series.groupby(x.code).pct_change(20)

    prev=g.close.shift(1)
    code=x.code.astype(str).str.zfill(6)
    # Conservative approximation for ordinary 10% names and 20% ChiNext/STAR names.
    x['limit_rate']=np.where(code.str.startswith(('300','301','688','689')),0.20,0.10)
    x['limit_up']=np.round(prev*(1+x.limit_rate),2)
    x['open_gap']=x.open/prev-1
    x['pinned_limit_open']=((x.open>=x.limit_up*0.9995)&(x.high<=x.open*1.0001)&(x.low>=x.open*0.9995))
    x=x[(x.date>='2022-01-04')&(x.date<='2026-03-19')].copy()

    # Signal is known on T close. This is deliberately broad and comes from the validated high-surge price-state family.
    x['signal']=(
        x['ma60_gap'].between(-0.604,-0.117) &
        x['ma20_slope'].between(-0.515,-0.0981) &
        x['r20'].between(-0.776,-0.12) &
        (x.close>=3)
    )
    parts.append(x)

df=pd.concat(parts,ignore_index=True).sort_values(['code','date']).reset_index(drop=True)
ng=df.groupby('code',group_keys=False)
for c in ['open','high','low','close','volume','amount','pctChg','open_gap','pinned_limit_open','limit_up']:
    df['next_'+c]=ng[c].shift(-1)
df['next_date']=ng.date.shift(-1)

rows=[]
for code,gp in df.groupby('code',sort=False):
    gp=gp.reset_index(drop=True)
    if len(gp)<22: continue
    signal_idx=np.flatnonzero(gp['signal'].fillna(False).to_numpy())
    for i in signal_idx:
        if i+20>=len(gp): continue
        row=gp.iloc[i]
        nextrow=gp.iloc[i+1]
        future_hi=gp.high.iloc[i+1:i+21]
        future_lo=gp.low.iloc[i+1:i+21]
        rows.append({
            'code':code,
            'signal_date':row.date,
            'next_date':nextrow.date,
            'signal_close':float(row.close),
            'next_open':float(nextrow.open),
            'next_high':float(nextrow.high),
            'next_low':float(nextrow.low),
            'next_close':float(nextrow.close),
            'next_open_gap':float(nextrow.open_gap) if pd.notna(nextrow.open_gap) else np.nan,
            'next_pinned_limit':bool(nextrow.pinned_limit_open),
            'future_high20':float(future_hi.max()),
            'future_low20':float(future_lo.min())
        })

res=pd.DataFrame(rows)
if res.empty:
    raise RuntimeError('No executable-entry signals were generated.')

# A: buy at T+1 open unless that open is pinned at the daily limit (NO_FILL).
res['A_fill']=(~res['next_pinned_limit']).astype(int)
res['A_entry']=np.where(res.A_fill.eq(1),res.next_open,np.nan)

# B: only buy at T+1 open when gap <= +5%; otherwise treat as NO_FILL in this daily-only test.
res['B_fill']=((~res['next_pinned_limit'])&(res.next_open_gap<=0.05)).astype(int)
res['B_entry']=np.where(res.B_fill.eq(1),res.next_open,np.nan)

# C: conservative pullback proxy. Place a limit 2% above the signal close;
# if T+1 low reaches it, count as filled at that target. No intraday sequencing is assumed.
res['C_target']=res.signal_close*1.02
res['C_fill']=((~res['next_pinned_limit'])&(res.next_low<=res.C_target)).astype(int)
res['C_entry']=np.where(res.C_fill.eq(1),res.C_target,np.nan)

for mode in 'ABC':
    entry=f'{mode}_entry'; fill=f'{mode}_fill'
    res[f'{mode}_mfe']=np.where(res[fill].eq(1),res.future_high20/res[entry]-1,np.nan)
    res[f'{mode}_mae']=np.where(res[fill].eq(1),res.future_low20/res[entry]-1,np.nan)
    for threshold,name in [(0.03,'mfe3'),(0.05,'mfe5'),(0.08,'mfe8')]:
        res[f'{mode}_{name}']=np.where(res[fill].eq(1),res[f'{mode}_mfe']>=threshold,np.nan)
    res[f'{mode}_fail5']=np.where(res[fill].eq(1),res[f'{mode}_mae']<=-0.05,np.nan)

summary={}
for mode,label in [('A','open_all_except_pinned'),('B','open_gap_cap_5pct'),('C','pullback_2pct')]:
    filled=res[res[f'{mode}_fill'].eq(1)]
    summary[label]={
        'signals':int(len(res)),
        'filled':int(len(filled)),
        'fill_rate':float(len(filled)/len(res)) if len(res) else None,
        'mfe3_rate':float(filled[f'{mode}_mfe3'].mean()) if len(filled) else None,
        'mfe5_rate':float(filled[f'{mode}_mfe5'].mean()) if len(filled) else None,
        'mfe8_rate':float(filled[f'{mode}_mfe8'].mean()) if len(filled) else None,
        'avg_mfe':float(filled[f'{mode}_mfe'].mean()) if len(filled) else None,
        'median_mfe':float(filled[f'{mode}_mfe'].median()) if len(filled) else None,
        'fail5_rate':float(filled[f'{mode}_fail5'].mean()) if len(filled) else None,
        'avg_mae':float(filled[f'{mode}_mae'].mean()) if len(filled) else None,
    }

(OUT/'executable_entry_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
res.to_parquet(OUT/'executable_entry_signals.parquet',index=False)
print(json.dumps(summary,ensure_ascii=False,indent=2))
