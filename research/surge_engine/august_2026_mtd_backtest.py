from __future__ import annotations

import concurrent.futures
import json
import math
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
CURRENT_START = pd.Timestamp('2026-03-20')
AS_OF = pd.Timestamp('2026-08-17')
SIG_START = pd.Timestamp('2026-08-01')
SIG_END = AS_OF

COMMISSION_BPS = 3.0
STAMP_BPS = 10.0
SLIPPAGE_BPS = 5.0
TRAIL_PCT = 0.03

FEATURES = [
    'r3','r5','r10','r20','ma20_gap','ma60_gap','ma20_slope',
    'dist20_high','dist60_high','dd20','atr14_pct','range_pct',
    'close_to_high','vol_ratio','upvol_ratio'
]
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
        keep = ['code','date','open','high','low','close','volume']
        missing = [c for c in keep if c not in x.columns]
        if missing:
            raise RuntimeError(f'{f} missing columns: {missing}')
        x = x[keep].copy()
        x['date'] = pd.to_datetime(x['date'])
        x['code'] = x['code'].astype(str).str.zfill(6)
        parts.append(x)
    df = pd.concat(parts, ignore_index=True)
    return df.drop_duplicates(['code','date']).sort_values(['code','date']).reset_index(drop=True)


def get_universe() -> pd.DataFrame:
    url = 'https://82.push2.eastmoney.com/api/qt/clist/get'
    params = {
        'pn': 1, 'pz': 50000, 'po': 1, 'np': 1,
        'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
        'fltt': 2, 'invt': 2, 'fid': 'f3',
        'fs': 'm:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048',
        'fields': 'f12,f13,f14',
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    obj = r.json()
    diff = (obj.get('data') or {}).get('diff') or {}
    rows = list(diff.values()) if isinstance(diff, dict) else diff
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError('Eastmoney returned empty stock universe')
    out = out.rename(columns={'f12':'code','f13':'market','f14':'name'})[['code','market','name']]
    out['code'] = out['code'].astype(str).str.zfill(6)
    out = out[~out['name'].astype(str).str.contains(r'\*?ST|退', regex=True, na=False)].copy()
    return out.drop_duplicates('code').reset_index(drop=True)


def fetch_one(code: str) -> Tuple[str, List[List[str]]]:
    market = '1' if code.startswith(('6','68','9')) else '0'
    if code.startswith(('4','8')):
        market = '0'
    url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
    params = {
        'secid': f'{market}.{code}',
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': 101, 'fqt': 1,
        'beg': '20260201', 'end': '20260817', 'lmt': 250,
    }
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if r.status_code == 429:
                raise RuntimeError('429')
            r.raise_for_status()
            klines = ((r.json().get('data') or {}).get('klines')) or []
            if klines:
                return code, [x.split(',') for x in klines]
            raise RuntimeError('empty klines')
        except Exception:
            if attempt == 3:
                return code, []
            time.sleep((0.5 * (2 ** attempt)) + random.random() * 0.3)
    return code, []


def fetch_extension(codes: List[str]) -> pd.DataFrame:
    rows = []
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(fetch_one, c) for c in codes]
        for fut in concurrent.futures.as_completed(futs):
            code, klines = fut.result()
            done += 1
            if done % 250 == 0:
                print(f'Fetched {done}/{len(codes)}', flush=True)
            for p in klines:
                if len(p) < 6:
                    continue
                rows.append({
                    'code': code,
                    'date': p[0], 'open': p[1], 'close': p[2], 'high': p[3],
                    'low': p[4], 'volume': p[5],
                })
    x = pd.DataFrame(rows)
    if x.empty:
        raise RuntimeError('No current extension data downloaded')
    for c in ['open','close','high','low','volume']:
        x[c] = pd.to_numeric(x[c], errors='coerce')
    x['date'] = pd.to_datetime(x['date'])
    x['code'] = x['code'].astype(str).str.zfill(6)
    return x.drop_duplicates(['code','date']).sort_values(['code','date']).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.sort_values(['code','date']).copy()
    g = x.groupby('code', group_keys=False)
    ma20 = g['close'].transform(lambda s: s.rolling(20).mean())
    ma60 = g['close'].transform(lambda s: s.rolling(60).mean())
    x['r3'] = g['close'].pct_change(3)
    x['r5'] = g['close'].pct_change(5)
    x['r10'] = g['close'].pct_change(10)
    x['r20'] = g['close'].pct_change(20)
    x['ma20_gap'] = x['close'] / ma20 - 1
    x['ma60_gap'] = x['close'] / ma60 - 1
    x['ma20_slope'] = ma20.groupby(x['code']).pct_change(20)
    x['dist20_high'] = x['close'] / g['close'].transform(lambda s: s.rolling(20).max()) - 1
    x['dist60_high'] = x['close'] / g['close'].transform(lambda s: s.rolling(60).max()) - 1
    roll20 = g['close'].transform(lambda s: s.rolling(20).max())
    x['dd20'] = x['close'] / roll20 - 1
    x['atr14_pct'] = (g['high'].transform(lambda s: s.rolling(14).max()) - g['low'].transform(lambda s: s.rolling(14).min())) / x['close']
    x['range_pct'] = (x['high'] - x['low']) / x['close']
    x['close_to_high'] = x['close'] / x['high'] - 1
    vol20 = g['volume'].transform(lambda s: s.rolling(20).mean())
    x['vol_ratio'] = x['volume'] / vol20
    x['upvol_ratio'] = np.where(x['close'] > g['close'].shift(1), x['volume'], 0) / vol20
    return x


def build_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    x = df[df['date'] <= TRAIN_END].copy().sort_values(['code','date'])
    rows = []
    for code, gp in x.groupby('code', sort=False):
        gp = gp.reset_index(drop=True)
        if len(gp) < 85:
            continue
        for i in range(60, len(gp) - 21):
            feat = gp.iloc[i]
            entry = float(gp.open.iloc[i+1]) if pd.notna(gp.open.iloc[i+1]) else np.nan
            future = gp.iloc[i+1:i+21]
            if not np.isfinite(entry) or entry <= 0 or future.empty:
                continue
            vals = {c: feat[c] for c in FEATURES}
            if any(pd.isna(v) or not np.isfinite(float(v)) for v in vals.values()):
                continue
            rows.append({
                'code': code, 'date': feat.date, 'entry': entry,
                **{c: float(v) for c,v in vals.items()},
                'future_mfe': float(future.high.max()/entry-1),
                'future_mae': float(future.low.min()/entry-1),
            })
    return pd.DataFrame(rows)


def best_bin(train: pd.DataFrame, feature: str) -> Tuple[float,float]:
    qs = train[feature].quantile(np.linspace(0,1,11)).drop_duplicates().to_numpy()
    if len(qs) < 4:
        raise RuntimeError(f'Not enough quantile points for {feature}')
    tmp = train.copy()
    tmp['_bin'] = pd.cut(tmp[feature], bins=qs, include_lowest=True, duplicates='drop')
    good = (tmp.future_mfe >= 0.03) & (tmp.future_mae > -0.05)
    g = tmp.assign(good=good).groupby('_bin', observed=True).agg(n=('good','size'), good=('good','mean')).reset_index()
    g['utility'] = g.good - 0.75 / np.sqrt(g.n)
    row = g.sort_values(['utility','n'], ascending=[False,False]).iloc[0]
    return float(row['_bin'].left), float(row['_bin'].right)


def learn_rules(train: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    bins = {f: best_bin(train, f) for f in set(sum(([a,b] for a,b in PAIR_RULES.values()), []))}
    rules = {}
    for sid,(f1,f2) in PAIR_RULES.items():
        lo1,hi1 = bins[f1]; lo2,hi2 = bins[f2]
        rules[sid] = {'factors':[f1,f2], 'ranges':{f1:[lo1,hi1], f2:[lo2,hi2]}}
    return rules


def signals_for(df: pd.DataFrame, rule: Dict[str, object]) -> pd.DataFrame:
    f1,f2 = rule['factors']
    r1_lo,r1_hi = rule['ranges'][f1]
    r2_lo,r2_hi = rule['ranges'][f2]
    x = df[(df.date >= SIG_START) & (df.date <= SIG_END)].copy()
    return x[x[f1].between(r1_lo,r1_hi) & x[f2].between(r2_lo,r2_hi)].sort_values(['date','code']).copy()


def net_return(entry: float, exit_px: float) -> float:
    buy = entry * (1 + SLIPPAGE_BPS/10000)
    sell = exit_px * (1 - SLIPPAGE_BPS/10000)
    buy_cost = buy * (1 + COMMISSION_BPS/10000)
    sell_net = sell * (1 - COMMISSION_BPS/10000 - STAMP_BPS/10000)
    return sell_net / buy_cost - 1


def simulate_trade(gp: pd.DataFrame, i: int) -> Dict[str, object]:
    if i+1 >= len(gp):
        return {'status':'pending_entry', 'entry_date':None, 'entry':None, 'exit_date':None, 'exit':None, 'ret':None, 'net_ret':None, 'reason':'pending_T1'}
    entry_date = gp.date.iloc[i+1]
    entry = float(gp.open.iloc[i+1])
    stop = entry * 0.95
    mode = 'hard_stop'
    peak = entry
    max_mfe = 0.0
    min_mae = 0.0
    last_date = entry_date
    last_close = float(gp.close.iloc[i+1])
    exit_px = last_close
    reason = None
    end = min(i + 20, len(gp) - 1)
    for j in range(i+1, end+1):
        hi = float(gp.high.iloc[j]); lo = float(gp.low.iloc[j]); cl = float(gp.close.iloc[j])
        last_date = gp.date.iloc[j]; last_close = cl
        peak = max(peak, hi)
        max_mfe = max(max_mfe, hi/entry-1)
        min_mae = min(min_mae, lo/entry-1)
        if lo <= stop:
            exit_px = stop; reason = mode; break
        if peak/entry-1 >= 0.05:
            stop = max(stop, peak*(1-TRAIL_PCT)); mode='trail'
        elif peak/entry-1 >= 0.03:
            stop = max(stop, entry*1.01); mode='protect_1pct'
        exit_px = cl
    if reason is None:
        if end == i + 20 and i + 20 < len(gp):
            reason = 't20'
        else:
            reason = 'open_as_of_2026-08-17'
    ret = exit_px/entry - 1 if reason != 'open_as_of_2026-08-17' else last_close/entry - 1
    net = net_return(entry, exit_px if reason != 'open_as_of_2026-08-17' else last_close)
    return {
        'status':'closed' if reason != 'open_as_of_2026-08-17' else 'open',
        'entry_date':entry_date.date().isoformat(),'entry':entry,
        'exit_date':last_date.date().isoformat() if reason != 'open_as_of_2026-08-17' else None,
        'exit':exit_px if reason != 'open_as_of_2026-08-17' else None,
        'last_close':last_close,
        'ret':float(ret),'net_ret':float(net),'reason':reason,
        'mfe':float(max_mfe),'mae':float(min_mae),'holding_days':int(end-i),
    }


def main():
    base = load_base()
    universe = get_universe()
    base_codes = set(base.code.unique())
    codes = [c for c in universe.code.tolist() if c in base_codes]
    print(f'Universe: {len(universe)}; using {len(codes)} codes with research history', flush=True)
    ext = fetch_extension(codes)
    df = pd.concat([base, ext], ignore_index=True).drop_duplicates(['code','date']).sort_values(['code','date']).reset_index(drop=True)
    df = add_features(df)
    train = build_training_rows(df)
    rules = learn_rules(train)
    print('Learned rules:', json.dumps(rules, ensure_ascii=False), flush=True)

    rows = []
    for sid, rule in rules.items():
        sig = signals_for(df, rule)
        for code, ss in sig.groupby('code', sort=False):
            gp = df[df.code.eq(code)].reset_index(drop=True)
            idx = {pd.Timestamp(d): i for i,d in enumerate(gp.date)}
            for _, s in ss.iterrows():
                i = idx.get(pd.Timestamp(s.date))
                if i is None:
                    continue
                trade = simulate_trade(gp, i)
                rows.append({'strategy':sid,'code':code,'signal_date':s.date.date().isoformat(), **trade})
    detail = pd.DataFrame(rows)
    if detail.empty:
        detail = pd.DataFrame(columns=['strategy','code','signal_date','status','entry_date','entry','exit_date','exit','last_close','ret','net_ret','reason','mfe','mae','holding_days'])
    # Current snapshot only includes signals in August. De-duplicate identical signal/code within a strategy.
    detail = detail.drop_duplicates(['strategy','code','signal_date']).sort_values(['signal_date','strategy','code'])
    summary = []
    for sid in rules:
        d = detail[detail.strategy.eq(sid)]
        closed = d[d.status.eq('closed')]
        open_d = d[d.status.eq('open')]
        pending = d[d.status.eq('pending_entry')]
        summary.append({
            'strategy': sid,
            'rule': rules[sid],
            'signals': int(len(d)),
            'closed_trades': int(len(closed)),
            'open_trades': int(len(open_d)),
            'pending_entries': int(len(pending)),
            'closed_win_rate_pct': float((closed.ret > 0).mean()*100) if len(closed) else None,
            'closed_avg_ret_pct': float(closed.ret.mean()*100) if len(closed) else None,
            'closed_avg_net_ret_pct': float(closed.net_ret.mean()*100) if len(closed) else None,
            'stop5_rate_pct': float((closed.reason.eq('hard_stop')).mean()*100) if len(closed) else None,
            'mfe3_rate_pct': float((closed.mfe >= 0.03).mean()*100) if len(closed) else None,
            'mfe5_rate_pct': float((closed.mfe >= 0.05).mean()*100) if len(closed) else None,
            'mfe8_rate_pct': float((closed.mfe >= 0.08).mean()*100) if len(closed) else None,
        })
    out = {
        'as_of':'2026-08-17',
        'signal_window':['2026-08-01','2026-08-17'],
        'note':'Full August is not yet available because 2026-08-18 onward is in the future. This is August month-to-date through 2026-08-17.',
        'cost_assumptions':{'commission_bps':COMMISSION_BPS,'stamp_tax_bps':STAMP_BPS,'slippage_bps':SLIPPAGE_BPS,'trail_pct':TRAIL_PCT},
        'rules':rules,
        'summary':summary,
    }
    (OUT/'august_2026_mtd_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    detail.to_csv(OUT/'august_2026_mtd_trade_detail.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    print('\nTRADE_DETAIL\n', detail.to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
