from __future__ import annotations

import concurrent.futures
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

DATA = Path('research/surge_engine/data')
OUT = Path('research/surge_engine/results')
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_END = pd.Timestamp('2026-03-19')
SIG_START = pd.Timestamp('2026-08-01')
SIG_END = pd.Timestamp('2026-08-17')

COMMISSION_BPS = 3.0
STAMP_BPS = 10.0
SLIPPAGE_BPS = 5.0
TRAIL_PCT = 0.03

FEATURES = ['r5', 'ma20_gap', 'ma60_gap', 'ma20_slope']
PAIR_RULES = {
    'A': ('r5', 'ma60_gap'),
    'B': ('ma20_slope', 'ma60_gap'),
    'C': ('ma20_gap', 'ma60_gap'),
}
HEADERS = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://quote.eastmoney.com/'}


def load_base() -> pd.DataFrame:
    parts = []
    for f in sorted(DATA.glob('kline_*.parquet')):
        x = pd.read_parquet(f)
        keep = ['code','date','open','high','low','close','volume']
        x = x[keep].copy()
        x['date'] = pd.to_datetime(x['date'])
        x['code'] = x['code'].astype(str).str.zfill(6)
        parts.append(x)
    return pd.concat(parts, ignore_index=True).drop_duplicates(['code','date']).sort_values(['code','date']).reset_index(drop=True)


def fetch_one_daily(code: str) -> Tuple[str, List[List[str]]]:
    market = '1' if code.startswith(('6','68','9')) else '0'
    if code.startswith(('4','8')):
        market = '0'
    url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
    params = {'secid': f'{market}.{code}', 'fields1':'f1,f2,f3,f4,f5,f6', 'fields2':'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61', 'klt':101, 'fqt':1, 'beg':'20260201', 'end':'20260817', 'lmt':250}
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=15)
            r.raise_for_status()
            klines = ((r.json().get('data') or {}).get('klines')) or []
            if klines:
                return code, [x.split(',') for x in klines]
        except Exception:
            pass
        time.sleep(0.6 * (2 ** attempt) + random.random() * 0.3)
    return code, []


def fetch_daily_extension(codes: List[str]) -> pd.DataFrame:
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as ex:
        for n,fut in enumerate(concurrent.futures.as_completed([ex.submit(fetch_one_daily,c) for c in codes]),1):
            code, klines = fut.result()
            if n % 250 == 0: print(f'daily fetched {n}/{len(codes)}', flush=True)
            for p in klines:
                if len(p) >= 6:
                    rows.append({'code':code,'date':p[0],'open':p[1],'close':p[2],'high':p[3],'low':p[4],'volume':p[5]})
    x=pd.DataFrame(rows)
    for c in ['open','close','high','low','volume']: x[c]=pd.to_numeric(x[c],errors='coerce')
    x['date']=pd.to_datetime(x['date']); x['code']=x['code'].astype(str).str.zfill(6)
    return x.drop_duplicates(['code','date']).sort_values(['code','date']).reset_index(drop=True)


def fetch_one_5m(code: str) -> Tuple[str, List[List[str]]]:
    market = '1' if code.startswith(('6','68','9')) else '0'
    if code.startswith(('4','8')): market='0'
    url='https://push2his.eastmoney.com/api/qt/stock/kline/get'
    params={'secid':f'{market}.{code}','fields1':'f1,f2,f3,f4,f5,f6','fields2':'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61','klt':5,'fqt':1,'beg':'20260801','end':'20260817','lmt':2500}
    for attempt in range(4):
        try:
            r=requests.get(url,params=params,headers=HEADERS,timeout=15); r.raise_for_status()
            klines=((r.json().get('data') or {}).get('klines')) or []
            if klines: return code,[x.split(',') for x in klines]
        except Exception: pass
        time.sleep(0.5*(2**attempt)+random.random()*0.2)
    return code,[]


def fetch_5m(codes: List[str]) -> pd.DataFrame:
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as ex:
        for n,fut in enumerate(concurrent.futures.as_completed([ex.submit(fetch_one_5m,c) for c in codes]),1):
            code,klines=fut.result()
            if n%250==0: print(f'5m fetched {n}/{len(codes)}', flush=True)
            for p in klines:
                if len(p)>=6:
                    rows.append({'code':code,'datetime':p[0],'open':p[1],'close':p[2],'high':p[3],'low':p[4],'volume':p[5]})
    x=pd.DataFrame(rows)
    if x.empty: return x
    for c in ['open','close','high','low','volume']: x[c]=pd.to_numeric(x[c],errors='coerce')
    x['datetime']=pd.to_datetime(x['datetime']); x['code']=x['code'].astype(str).str.zfill(6)
    return x.sort_values(['code','datetime']).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x=df.sort_values(['code','date']).copy(); g=x.groupby('code',group_keys=False)
    ma20=g['close'].transform(lambda s:s.rolling(20).mean()); ma60=g['close'].transform(lambda s:s.rolling(60).mean())
    x['r5']=g['close'].pct_change(5); x['ma20_gap']=x['close']/ma20-1; x['ma60_gap']=x['close']/ma60-1; x['ma20_slope']=ma20.groupby(x['code']).pct_change(20)
    return x


def best_bin(train: pd.DataFrame, feature: str) -> Tuple[float,float]:
    qs=train[feature].quantile(np.linspace(0,1,11)).drop_duplicates().to_numpy(); tmp=train.copy(); tmp['_bin']=pd.cut(tmp[feature],bins=qs,include_lowest=True,duplicates='drop')
    good=(tmp.future_mfe>=0.03)&(tmp.future_mae>-0.05)
    g=tmp.assign(good=good).groupby('_bin',observed=True).agg(n=('good','size'),good=('good','mean')).reset_index(); g['utility']=g.good-0.75/np.sqrt(g.n)
    row=g.sort_values(['utility','n'],ascending=[False,False]).iloc[0]; return float(row['_bin'].left),float(row['_bin'].right)


def learn_rules(df: pd.DataFrame) -> Dict[str,Dict[str,object]]:
    x=df[df.date<=TRAIN_END].copy(); rows=[]
    for code,gp0 in x.groupby('code',sort=False):
        gp=gp0.reset_index(drop=True)
        if len(gp)<90: continue
        for i in range(60,len(gp)-21):
            feat=gp.iloc[i]; entry=float(gp.open.iloc[i+1]); future=gp.iloc[i+1:i+21]
            if not np.isfinite(entry) or future.empty: continue
            vals={f:feat[f] for f in FEATURES}
            if any(pd.isna(v) or not np.isfinite(float(v)) for v in vals.values()): continue
            rows.append({'code':code,'date':feat.date,'entry':entry,**{f:float(vals[f]) for f in FEATURES},'future_mfe':float(future.high.max()/entry-1),'future_mae':float(future.low.min()/entry-1)})
    train=pd.DataFrame(rows)
    bins={f:best_bin(train,f) for f in set(sum(([a,b] for a,b in PAIR_RULES.values()),[]))}
    return {sid:{'factors':[f1,f2],'ranges':{f1:[*bins[f1]],f2:[*bins[f2]]}} for sid,(f1,f2) in PAIR_RULES.items()}


def match(v: float, r: List[float]) -> bool:
    return np.isfinite(v) and v>=r[0] and v<=r[1]


def framework_a(sig: pd.DataFrame) -> Dict[str,object]:
    # D close signal -> D+1 open buy -> D+2 earliest sell.
    out=[]
    for _,s in sig.iterrows():
        code=s.code; d=s.signal_date; gp=DAILY[DAILY.code.eq(code)].sort_values('date').reset_index(drop=True); pos=gp.index[gp.date.eq(d)]
        if len(pos)==0 or pos[0]+2>=len(gp): continue
        i=int(pos[0]); entry_date=gp.date.iloc[i+1]; entry=float(gp.open.iloc[i+1]); stop=entry*0.95; peak=entry; reason=None; exit_px=None; exit_date=None
        for j in range(i+2,len(gp)):
            hi=float(gp.high.iloc[j]); lo=float(gp.low.iloc[j]); cl=float(gp.close.iloc[j]); peak=max(peak,hi)
            if lo<=stop: exit_px=stop; exit_date=gp.date.iloc[j]; reason='hard_stop'; break
            if peak/entry-1>=0.05: stop=max(stop,peak*(1-TRAIL_PCT))
            elif peak/entry-1>=0.03: stop=max(stop,entry*1.01)
            if j-i>=20: exit_px=cl; exit_date=gp.date.iloc[j]; reason='t20'; break
        if exit_px is None:
            continue
        out.append({'framework':'A','strategy':s.strategy,'code':code,'signal_date':d.date().isoformat(),'entry_date':entry_date.date().isoformat(),'entry':entry,'exit_date':exit_date.date().isoformat(),'exit':float(exit_px),'net_ret':net_return(entry,exit_px),'reason':reason})
    return out


def framework_c(sig: pd.DataFrame) -> Dict[str,object]:
    # D close signal -> D close buy -> D+1 earliest sell, with same daily trigger logic.
    out=[]
    for _,s in sig.iterrows():
        code=s.code; d=s.signal_date; gp=DAILY[DAILY.code.eq(code)].sort_values('date').reset_index(drop=True); pos=gp.index[gp.date.eq(d)]
        if len(pos)==0 or pos[0]+1>=len(gp): continue
        i=int(pos[0]); entry=float(gp.close.iloc[i]); stop=entry*0.95; peak=entry
        j=i+1; hi=float(gp.high.iloc[j]); lo=float(gp.low.iloc[j]); cl=float(gp.close.iloc[j]); peak=max(peak,hi)
        if lo<=stop: exit_px=stop; reason='hard_stop'
        elif peak/entry-1>=0.05: exit_px=stop if peak*(1-TRAIL_PCT)<=hi else cl; reason='trail' if exit_px!=stop else 'trail'
        elif peak/entry-1>=0.03: exit_px=max(entry*1.01,cl); reason='protect_1pct' if cl>=entry*1.01 else 'hold'
        else: exit_px=cl; reason='d1_close'
        out.append({'framework':'C','strategy':s.strategy,'code':code,'signal_date':d.date().isoformat(),'entry_date':d.date().isoformat(),'entry':entry,'exit_date':gp.date.iloc[j].date().isoformat(),'exit':float(exit_px),'net_ret':net_return(entry,exit_px),'reason':reason})
    return out


def framework_b(rules: Dict[str,Dict[str,object]], five: pd.DataFrame) -> Dict[str,object]:
    if five.empty: return []
    # D 14:30 signal -> D 14:55 buy proxy -> D+1 earliest sell.
    # Factors use 14:30 price plus completed prior daily closes.
    daily_by_code={c:g.sort_values('date').reset_index(drop=True) for c,g in DAILY.groupby('code')}
    out=[]
    x=five[five.datetime.dt.strftime('%H:%M:%S').isin(['14:30:00','14:35:00','14:40:00','14:45:00','14:50:00','14:55:00'])].copy()
    for code,g5 in x.groupby('code'):
        gp=daily_by_code.get(code)
        if gp is None: continue
        for day in sorted(pd.to_datetime(g5.datetime.dt.date).unique()):
            day=pd.Timestamp(day); bars=g5[g5.datetime.dt.normalize().eq(day.normalize())].sort_values('datetime')
            b1430=bars[bars.datetime.dt.strftime('%H:%M:%S').eq('14:30:00')]
            b1455=bars[bars.datetime.dt.strftime('%H:%M:%S').eq('14:55:00')]
            if b1430.empty or b1455.empty: continue
            p=float(b1430.close.iloc[-1]); prior=gp[gp.date.lt(day)].copy()
            if len(prior)<60: continue
            ma20=(prior.close.tail(19).sum()+p)/20; ma60=(prior.close.tail(59).sum()+p)/60
            r5=p/float(prior.close.iloc[-5])-1; ma20_gap=p/ma20-1; ma60_gap=p/ma60-1
            ma20_prev=prior.close.tail(20).mean(); prior20=prior.close.iloc[-39:-19].mean() if len(prior)>=39 else np.nan; ma20_slope=ma20/prior20-1 if prior20 and np.isfinite(prior20) else np.nan
            feats={'r5':r5,'ma20_gap':ma20_gap,'ma60_gap':ma60_gap,'ma20_slope':ma20_slope}
            for sid,rule in rules.items():
                f1,f2=rule['factors'];
                if not (match(feats[f1],rule['ranges'][f1]) and match(feats[f2],rule['ranges'][f2])): continue
                entry=float(b1455.close.iloc[-1]); nextd=gp[gp.date.gt(day)].head(1)
                if nextd.empty: continue
                nd=nextd.iloc[0]; out.append({'framework':'B','strategy':sid,'code':code,'signal_date':day.date().isoformat(),'entry_date':day.date().isoformat(),'entry':entry,'exit_date':nd.date.date().isoformat(),'exit':float(nd.close),'net_ret':net_return(entry,float(nd.close)),'reason':'d1_close_proxy'})
    return out


def net_return(entry: float, exit_px: float) -> float:
    buy=entry*(1+SLIPPAGE_BPS/10000); sell=exit_px*(1-SLIPPAGE_BPS/10000); return (sell*(1-COMMISSION_BPS/10000-STAMP_BPS/10000))/(buy*(1+COMMISSION_BPS/10000))-1


def main():
    base=load_base(); codes=sorted(base.code.unique().tolist()); print(f'Using all research-history codes: {len(codes)}',flush=True)
    ext=fetch_daily_extension(codes); daily=pd.concat([base,ext],ignore_index=True).drop_duplicates(['code','date']).sort_values(['code','date']).reset_index(drop=True); daily=add_features(daily)
    global DAILY; DAILY=daily
    rules=learn_rules(daily); print('rules',json.dumps(rules,ensure_ascii=False),flush=True)
    # Daily signal matrix for frameworks A/C.
    sigrows=[]
    x=daily[(daily.date>=SIG_START)&(daily.date<=SIG_END)].copy()
    for sid,rule in rules.items():
        f1,f2=rule['factors']; mask=x[f1].between(*rule['ranges'][f1]) & x[f2].between(*rule['ranges'][f2]); tmp=x[mask][['code','date']].copy(); tmp['strategy']=sid; tmp=tmp.rename(columns={'date':'signal_date'}); sigrows.append(tmp)
    sig=pd.concat(sigrows,ignore_index=True).drop_duplicates(['strategy','code','signal_date'])
    a=framework_a(sig); c=framework_c(sig); five=fetch_5m(codes); b=framework_b(rules,five)
    detail=pd.DataFrame(a+b+c)
    if detail.empty: detail=pd.DataFrame(columns=['framework','strategy','code','signal_date','entry_date','entry','exit_date','exit','net_ret','reason'])
    rows=[]
    for fw in ['A','B','C']:
        d=detail[detail.framework.eq(fw)]; rows.append({'framework':fw,'trades':int(len(d)),'win_rate_pct':float((d.net_ret>0).mean()*100) if len(d) else None,'avg_net_ret_pct':float(d.net_ret.mean()*100) if len(d) else None,'median_net_ret_pct':float(d.net_ret.median()*100) if len(d) else None,'stop5_rate_pct':float(d.reason.eq('hard_stop').mean()*100) if len(d) else None})
    summary={'as_of':'2026-08-17','signal_window':['2026-08-01','2026-08-17'],'note':'Framework B uses 5-minute Eastmoney data; 14:30 signal and 14:55 close proxy for entry. A and C use daily data and strict T+1 exit timing. This is an execution-framework comparison, not a finalized live rule.','rules':rules,'framework_summary':rows}
    (OUT/'august_2026_three_framework_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    detail.to_csv(OUT/'august_2026_three_framework_detail.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print(detail.to_string(index=False),flush=True)

if __name__=='__main__': main()
