from pathlib import Path
import pandas as pd, numpy as np, json

DATA=Path('research/surge_engine/data'); OUT=Path('research/surge_engine/results'); OUT.mkdir(parents=True,exist_ok=True)

# Use the strongest validated context rule family as signal proxy, then compare executable entry modes.
# Signal is formed on T close; entry happens on T+1. We only use daily OHLC available in the public dataset.
# This is intentionally conservative: a limit-up open that remains pinned at the high is marked NO_FILL.
parts=[]
for f in sorted(DATA.glob('kline_*.parquet')):
    x=pd.read_parquet(f,columns=['code','date','open','high','low','close','volume','amount','pctChg'])
    x['date']=pd.to_datetime(x['date']); x=x.sort_values(['code','date'])
    g=x.groupby('code',group_keys=False)
    x['r5']=g.close.pct_change(5); x['r20']=g.close.pct_change(20)
    ma20=g.close.transform(lambda s:s.rolling(20).mean()); ma60=g.close.transform(lambda s:s.rolling(60).mean())
    x['ma20_gap']=x.close/ma20-1; x['ma60_gap']=x.close/ma60-1
    x['ma20_slope']=g.close.transform(lambda s:s.rolling(20).mean()).groupby(x.code).pct_change(20)
    prev=g.close.shift(1)
    # Approximate price-limit bands using 10% for regular main-board names; 20% for ChiNext/STAR.
    # ST/other special limits are flagged as ambiguous and treated conservatively.
    code=x.code.astype(str).str.zfill(6)
    x['limit_rate']=np.where(code.str.startswith(('300','301','688','689')),0.20,0.10)
    x['limit_up']=np.round(prev*(1+x.limit_rate)+1e-8,2)
    x['open_gap']=x.open/prev-1
    x['gap_to_high']=x.high/x.open-1
    x['pinned_limit_open']=((x.open>=x.limit_up*0.9995)&(x.high<=x.open*1.0001)&(x.low>=x.open*0.9995))
    x=x[(x.date>='2022-01-04')&(x.date<='2026-03-19')].copy()
    parts.append(x)
df=pd.concat(parts,ignore_index=True).sort_values(['code','date'])

# Shift next-day execution fields onto signal date.
ng=df.groupby('code',group_keys=False)
for c in ['open','high','low','close','volume','amount','pctChg','open_gap','gap_to_high','pinned_limit_open','limit_up']:
    df['next_'+c]=ng[c].shift(-1)
df['next_date']=ng.date.shift(-1)

# Reuse a simple high-surge signal built from the strongest price-state region discovered earlier.
# These thresholds are kept broad on purpose; the final engine will be re-fit after entry-mode comparison.
signal=(
    df['ma60_gap'].between(-0.604,-0.117) &
    df['ma20_slope'].between(-0.515,-0.0981) &
    df['r20'].between(-0.776,-0.12)
)
# liquidity/price sanity
signal &= (df.close>=3) & (df.amount.rolling(20).mean().groupby(df.code).transform('last')>0) if False else True
sig=df.loc[signal].copy()

# Only signals with a real next trading session.
sig=sig[sig.next_date.notna()].copy()

# Approximate theoretical MFE after each candidate entry using next-day and future daily highs from existing data.
# For each signal, we collect the next 20 sessions' highs/lows and use entry price for MFE/MAE.
def forward_metrics(group):
    highs=group.high.to_numpy(); lows=group.low.to_numpy(); opens=group.open.to_numpy()
    dates=group.date.to_numpy(); idx={d:i for i,d in enumerate(dates)}
    out=[]
    for i,row in group.iterrows():
        j=idx.get(row.date)
        if j is None or j+20>=len(group): continue
        future_hi=highs[j+1:j+21]; future_lo=lows[j+1:j+21]
        for mode,entry,filled in []: pass
    return out

# Create per-signal future high/low arrays efficiently by positional loop on each stock.
rows=[]
for code,gp in df.groupby('code',sort=False):
    gp=gp.reset_index(drop=True)
    if len(gp)<22: continue
    for i in np.flatnonzero(signal.loc[gp.index].to_numpy() if len(signal)==len(df) else np.zeros(len(gp),dtype=bool)):
        if i+20>=len(gp): continue
        row=gp.iloc[i]
        nextrow=gp.iloc[i+1]
        rows.append({
            'code':code,'signal_date':row.date,'next_date':nextrow.date,
            'signal_close':row.close,'next_open':nextrow.open,'next_high':nextrow.high,'next_low':nextrow.low,'next_close':nextrow.close,
            'next_open_gap':nextrow.open_gap,'next_pinned_limit':bool(nextrow.pinned_limit_open),
            'future_high20':float(gp.high.iloc[i+1:i+21].max()),'future_low20':float(gp.low.iloc[i+1:i+21].min())
        })
res=pd.DataFrame(rows)

# Entry modes. No intraday data is available, so VWAP/5-min are intentionally not fabricated.
# A = next open, except pinned limit-up open => NO_FILL.
res['A_fill']=(~res.next_pinned_limit).astype(int)
res['A_entry']=np.where(res.A_fill.eq(1),res.next_open,np.nan)
# B = conservative cap: buy at next open only when gap <= +5%; otherwise wait/not filled in this daily-only test.
res['B_fill']=((~res.next_pinned_limit)&(res.next_open_gap<=0.05)).astype(int)
res['B_entry']=np.where(res.B_fill.eq(1),res.next_open,np.nan)
# C = pullback proxy: if next-day low revisits within 2% of previous close, assume a limit order at prev close*1.02 is executable.
res['C_target']=res.signal_close*1.02
res['C_fill']=((~res.next_pinned_limit)&(res.next_low<=res.C_target)).astype(int)
res['C_entry']=np.where(res.C_fill.eq(1),res.C_target,np.nan)

for mode in 'ABC':
    entry=f'{mode}_entry'; fill=f'{mode}_fill'
    res[f'{mode}_mfe']=np.where(res[fill].eq(1),res.future_high20/res[entry]-1,np.nan)
    res[f'{mode}_mae']=np.where(res[fill].eq(1),res.future_low20/res[entry]-1,np.nan)
    res[f'{mode}_mfe3']=np.where(res[fill].eq(1),res[f'{mode}_mfe']>=0.03,np.nan)
    res[f'{mode}_mfe5']=np.where(res[fill].eq(1),res[f'{mode}_mfe']>=0.05,np.nan)
    res[f'{mode}_mfe8']=np.where(res[fill].eq(1),res[f'{mode}_mfe']>=0.08,np.nan)
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
