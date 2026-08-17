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
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/151.0 Safari/537.36',
    'Referer': 'https://quote.eastmoney.com/',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}


def load_base() -> pd.DataFrame:
    parts = []
    for f in sorted(DATA.glob('kline_*.parquet')):
        x = pd.read_parquet(f)
        keep = ['code', 'date', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in keep if c not in x.columns]
        if missing:
            raise RuntimeError(f'{f} missing columns: {missing}')
        x = x[keep].copy()
        x['date'] = pd.to_datetime(x['date'])
        x['code'] = x['code'].astype(str).str.zfill(6)
        parts.append(x)
    if not parts:
        raise RuntimeError('No parquet research data found')
    return pd.concat(parts, ignore_index=True).drop_duplicates(['code', 'date']).sort_values(['code', 'date']).reset_index(drop=True)


def get_universe() -> set[str]:
    url = 'https://82.push2.eastmoney.com/api/qt/clist/get'
    params = {
        'pn': 1, 'pz': 50000, 'po': 1, 'np': 1,
        'ut': 'bd1d9ddb04089700cf9c27f6f7426281', 'fltt': 2, 'invt': 2,
        'fid': 'f3', 'fs': 'm:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048',
        'fields': 'f12,f14',
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    diff = (r.json().get('data') or {}).get('diff') or {}
    rows = list(diff.values()) if isinstance(diff, dict) else diff
    out = pd.DataFrame(rows)
    if out.empty:
        return set()
    out = out.rename(columns={'f12': 'code', 'f14': 'name'})
    out['code'] = out['code'].astype(str).str.zfill(6)
    out = out[~out['name'].astype(str).str.contains(r'\*?ST|退', regex=True, na=False)]
    return set(out['code'].drop_duplicates().tolist())


def fetch_one_daily(code: str) -> Tuple[str, List[List[str]]]:
    market = '1' if code.startswith(('6', '68', '9')) else '0'
    if code.startswith(('4', '8')):
        market = '0'
    url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
    params = {
        'secid': f'{market}.{code}', 'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': 101, 'fqt': 1, 'beg': '20260201', 'end': '20260817', 'lmt': 250,
    }
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
    rows: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        futs = [ex.submit(fetch_one_daily, c) for c in codes]
        for n, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            code, klines = fut.result()
            if n % 500 == 0:
                print(f'daily fetched {n}/{len(codes)}', flush=True)
            for p in klines:
                if len(p) >= 6:
                    rows.append({'code': code, 'date': p[0], 'open': p[1], 'close': p[2], 'high': p[3], 'low': p[4], 'volume': p[5]})
    if not rows:
        raise RuntimeError('No daily extension data returned from Eastmoney')
    x = pd.DataFrame(rows, columns=['code', 'date', 'open', 'close', 'high', 'low', 'volume'])
    for c in ['open', 'close', 'high', 'low', 'volume']:
        x[c] = pd.to_numeric(x[c], errors='coerce')
    x['date'] = pd.to_datetime(x['date'])
    x['code'] = x['code'].astype(str).str.zfill(6)
    return x.drop_duplicates(['code', 'date']).sort_values(['code', 'date']).reset_index(drop=True)


def fetch_one_5m(code: str) -> Tuple[str, List[List[str]]]:
    market = '1' if code.startswith(('6', '68', '9')) else '0'
    if code.startswith(('4', '8')):
        market = '0'
    url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
    params = {
        'secid': f'{market}.{code}', 'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': 5, 'fqt': 1, 'beg': '20260801', 'end': '20260817', 'lmt': 2500,
    }
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=15)
            r.raise_for_status()
            klines = ((r.json().get('data') or {}).get('klines')) or []
            if klines:
                return code, [x.split(',') for x in klines]
        except Exception:
            pass
        time.sleep(0.4 * (2 ** attempt) + random.random() * 0.2)
    return code, []


def fetch_5m(codes: List[str]) -> pd.DataFrame:
    rows: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        futs = [ex.submit(fetch_one_5m, c) for c in codes]
        for n, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            code, klines = fut.result()
            if n % 500 == 0:
                print(f'5m fetched {n}/{len(codes)}', flush=True)
            for p in klines:
                if len(p) >= 6:
                    rows.append({'code': code, 'datetime': p[0], 'open': p[1], 'close': p[2], 'high': p[3], 'low': p[4], 'volume': p[5]})
    if not rows:
        raise RuntimeError('No 5m data returned from Eastmoney')
    x = pd.DataFrame(rows, columns=['code', 'datetime', 'open', 'close', 'high', 'low', 'volume'])
    for c in ['open', 'close', 'high', 'low', 'volume']:
        x[c] = pd.to_numeric(x[c], errors='coerce')
    x['datetime'] = pd.to_datetime(x['datetime'])
    x['code'] = x['code'].astype(str).str.zfill(6)
    return x.sort_values(['code', 'datetime']).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.sort_values(['code', 'date']).copy()
    g = x.groupby('code', group_keys=False)
    ma20 = g['close'].transform(lambda s: s.rolling(20).mean())
    ma60 = g['close'].transform(lambda s: s.rolling(60).mean())
    x['r5'] = g['close'].pct_change(5)
    x['ma20_gap'] = x['close'] / ma20 - 1
    x['ma60_gap'] = x['close'] / ma60 - 1
    x['ma20_slope'] = ma20.groupby(x['code']).pct_change(20)
    return x


def best_bin(train: pd.DataFrame, feature: str) -> Tuple[float, float]:
    qs = train[feature].quantile(np.linspace(0, 1, 11)).drop_duplicates().to_numpy()
    tmp = train.copy()
    tmp['_bin'] = pd.cut(tmp[feature], bins=qs, include_lowest=True, duplicates='drop')
    good = (tmp.future_mfe >= 0.03) & (tmp.future_mae > -0.05)
    g = tmp.assign(good=good).groupby('_bin', observed=True).agg(n=('good', 'size'), good=('good', 'mean')).reset_index()
    g['utility'] = g.good - 0.75 / np.sqrt(g.n)
    row = g.sort_values(['utility', 'n'], ascending=[False, False]).iloc[0]
    return float(row['_bin'].left), float(row['_bin'].right)


def learn_rules(df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    x = df[df.date <= TRAIN_END].copy()
    rows = []
    for code, gp0 in x.groupby('code', sort=False):
        gp = gp0.reset_index(drop=True)
        if len(gp) < 90:
            continue
        for i in range(60, len(gp) - 21):
            feat = gp.iloc[i]
            entry = float(gp.open.iloc[i + 1])
            future = gp.iloc[i + 1:i + 21]
            vals = {f: feat[f] for f in FEATURES}
            if not future.empty and np.isfinite(entry) and entry > 0 and all(pd.notna(v) and np.isfinite(float(v)) for v in vals.values()):
                rows.append({'code': code, 'date': feat.date, 'entry': entry, **{f: float(vals[f]) for f in FEATURES}, 'future_mfe': float(future.high.max() / entry - 1), 'future_mae': float(future.low.min() / entry - 1)})
    train = pd.DataFrame(rows)
    if train.empty:
        raise RuntimeError('Training rows empty')
    bins = {f: best_bin(train, f) for f in set(sum(([a, b] for a, b in PAIR_RULES.values()), []))}
    return {sid: {'factors': [f1, f2], 'ranges': {f1: [*bins[f1]], f2: [*bins[f2]]}} for sid, (f1, f2) in PAIR_RULES.items()}


def net_return(entry: float, exit_px: float) -> float:
    buy = entry * (1 + SLIPPAGE_BPS / 10000)
    sell = exit_px * (1 - SLIPPAGE_BPS / 10000)
    return (sell * (1 - COMMISSION_BPS / 10000 - STAMP_BPS / 10000)) / (buy * (1 + COMMISSION_BPS / 10000)) - 1


def framework_a(sig: pd.DataFrame, daily: pd.DataFrame) -> list[dict]:
    out=[]
    for _, s in sig.iterrows():
        gp = daily[daily.code.eq(s.code)].sort_values('date').reset_index(drop=True)
        pos = gp.index[gp.date.eq(s.signal_date)]
        if len(pos) == 0 or int(pos[0]) + 2 >= len(gp):
            continue
        i=int(pos[0]); entry_date=gp.date.iloc[i+1]; entry=float(gp.open.iloc[i+1])
        stop=entry*0.95; peak=entry; exit_px=None; exit_date=None; reason=None
        for j in range(i+2, min(len(gp), i+21)):
            hi=float(gp.high.iloc[j]); lo=float(gp.low.iloc[j]); cl=float(gp.close.iloc[j]); peak=max(peak,hi)
            if lo<=stop:
                exit_px=stop; exit_date=gp.date.iloc[j]; reason='hard_stop'; break
            if peak/entry-1>=0.05: stop=max(stop, peak*(1-TRAIL_PCT))
            elif peak/entry-1>=0.03: stop=max(stop, entry*1.01)
            if j-i>=20:
                exit_px=cl; exit_date=gp.date.iloc[j]; reason='t20'; break
        if exit_px is None:
            continue
        out.append({'framework':'A','strategy':s.strategy,'code':s.code,'signal_date':s.signal_date.date().isoformat(),'entry_date':entry_date.date().isoformat(),'entry':entry,'exit_date':exit_date.date().isoformat(),'exit':float(exit_px),'net_ret':net_return(entry,float(exit_px)),'reason':reason})
    return out


def framework_c(sig: pd.DataFrame, daily: pd.DataFrame) -> list[dict]:
    out=[]
    for _, s in sig.iterrows():
        gp=daily[daily.code.eq(s.code)].sort_values('date').reset_index(drop=True)
        pos=gp.index[gp.date.eq(s.signal_date)]
        if len(pos)==0 or int(pos[0])+1>=len(gp): continue
        i=int(pos[0]); entry=float(gp.close.iloc[i]); j=i+1
        hi=float(gp.high.iloc[j]); lo=float(gp.low.iloc[j]); cl=float(gp.close.iloc[j])
        # C is deliberately simple: buy at D close, sell at D+1 close, T+1 respected.
        exit_px=entry*0.95 if lo <= entry*0.95 else cl
        reason='hard_stop' if lo <= entry*0.95 else 'd1_close'
        out.append({'framework':'C','strategy':s.strategy,'code':s.code,'signal_date':s.signal_date.date().isoformat(),'entry_date':s.signal_date.date().isoformat(),'entry':entry,'exit_date':gp.date.iloc[j].date().isoformat(),'exit':float(exit_px),'net_ret':net_return(entry,float(exit_px)),'reason':reason})
    return out


def framework_b(rules: Dict[str, Dict[str, object]], five: pd.DataFrame, daily: pd.DataFrame) -> list[dict]:
    daily_by_code={c:g.sort_values('date').reset_index(drop=True) for c,g in daily.groupby('code')}
    out=[]
    x=five[five.datetime.dt.strftime('%H:%M:%S').isin(['14:30:00','14:35:00','14:40:00','14:45:00','14:50:00','14:55:00'])].copy()
    for code,g5 in x.groupby('code'):
        gp=daily_by_code.get(code)
        if gp is None: continue
        for day_raw in sorted(pd.to_datetime(g5.datetime.dt.date).unique()):
            day=pd.Timestamp(day_raw); bars=g5[g5.datetime.dt.normalize().eq(day.normalize())].sort_values('datetime')
            b1430=bars[bars.datetime.dt.strftime('%H:%M:%S').eq('14:30:00')]; b1455=bars[bars.datetime.dt.strftime('%H:%M:%S').eq('14:55:00')]
            if b1430.empty or b1455.empty: continue
            prior=gp[gp.date.lt(day)]
            if len(prior)<60: continue
            p=float(b1430.close.iloc[-1])
            ma20=(prior.close.tail(19).sum()+p)/20; ma60=(prior.close.tail(59).sum()+p)/60
            r5=p/float(prior.close.iloc[-5])-1
            prior20=prior.close.iloc[-39:-19].mean(); ma20_slope=ma20/prior20-1 if len(prior)>=39 and np.isfinite(prior20) and prior20!=0 else np.nan
            feats={'r5':r5,'ma20_gap':p/ma20-1,'ma60_gap':p/ma60-1,'ma20_slope':ma20_slope}
            nextd=gp[gp.date.gt(day)].head(1)
            if nextd.empty: continue
            nd=nextd.iloc[0]
            for sid,rule in rules.items():
                f1,f2=rule['factors']
                if not (np.isfinite(feats[f1]) and np.isfinite(feats[f2])): continue
                if not (rule['ranges'][f1][0] <= feats[f1] <= rule['ranges'][f1][1] and rule['ranges'][f2][0] <= feats[f2] <= rule['ranges'][f2][1]): continue
                entry=float(b1455.close.iloc[-1]); exit_px=float(nd.close)
                out.append({'framework':'B','strategy':sid,'code':code,'signal_date':day.date().isoformat(),'entry_date':day.date().isoformat(),'entry':entry,'exit_date':nd.date.date().isoformat(),'exit':exit_px,'net_ret':net_return(entry,exit_px),'reason':'d1_close'})
    return out


def summarize(detail: pd.DataFrame) -> list[dict]:
    rows=[]
    for fw in ['A','B','C']:
        d=detail[detail.framework.eq(fw)]
        rows.append({'framework':fw,'trades':int(len(d)),'win_rate_pct':float((d.net_ret>0).mean()*100) if len(d) else None,'avg_net_ret_pct':float(d.net_ret.mean()*100) if len(d) else None,'median_net_ret_pct':float(d.net_ret.median()*100) if len(d) else None,'stop5_rate_pct':float(d.reason.eq('hard_stop').mean()*100) if len(d) else None,'avg_mfe_pct':None})
    return rows


def main():
    base=load_base(); active=get_universe(); codes=sorted(set(base.code).intersection(active)) if active else sorted(base.code.unique().tolist()); print(f'Using active universe with research history: {len(codes)}',flush=True)
    ext=fetch_daily_extension(codes); daily=pd.concat([base[base.code.isin(codes)],ext],ignore_index=True).drop_duplicates(['code','date']).sort_values(['code','date']).reset_index(drop=True); daily=add_features(daily)
    rules=learn_rules(daily); print('rules',json.dumps(rules,ensure_ascii=False),flush=True)
    x=daily[(daily.date>=SIG_START)&(daily.date<=SIG_END)]
    sigrows=[]
    for sid,rule in rules.items():
        f1,f2=rule['factors']; m=x[f1].between(*rule['ranges'][f1]) & x[f2].between(*rule['ranges'][f2]); t=x.loc[m,['code','date']].copy(); t['strategy']=sid; t=t.rename(columns={'date':'signal_date'}); sigrows.append(t)
    sig=pd.concat(sigrows,ignore_index=True).drop_duplicates(['strategy','code','signal_date']) if sigrows else pd.DataFrame(columns=['strategy','code','signal_date'])
    print(f'daily signals={len(sig)}',flush=True)
    a=framework_a(sig,daily); c=framework_c(sig,daily)
    five=fetch_5m(codes)
    b=framework_b(rules,five,daily)
    detail=pd.DataFrame(a+b+c)
    if detail.empty: detail=pd.DataFrame(columns=['framework','strategy','code','signal_date','entry_date','entry','exit_date','exit','net_ret','reason'])
    summary={'as_of':'2026-08-17','signal_window':['2026-08-01','2026-08-17'],'rules':rules,'framework_summary':summarize(detail),'trade_count':int(len(detail)),'note':'A: D close signal -> D+1 open buy -> D+2 earliest sell. B: D 14:30 signal -> D 14:55 buy proxy -> D+1 close sell. C: D close buy -> D+1 close sell. All respect T+1. Framework B uses completed 5-minute bars only.'}
    (OUT/'august_2026_three_framework_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    detail.to_csv(OUT/'august_2026_three_framework_detail.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True); print(detail.to_string(index=False),flush=True)

if __name__=='__main__': main()
