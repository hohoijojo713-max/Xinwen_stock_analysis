#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd

DATA_DIR = Path('research/surge_engine/data')
OUT = Path('research/surge_engine/results')
OUT.mkdir(parents=True, exist_ok=True)

files = sorted(DATA_DIR.glob('kline_*.parquet'))
if not files:
    raise SystemExit('No kline parquet files found')

rows = 0
stocks = set()
global_min = None
global_max = None
duplicates = 0
bad_ohlc = 0
bad_volume = 0
reports = []

for f in files:
    df = pd.read_parquet(f, columns=['code','date','open','high','low','close','volume','amount','turn','pctChg'])
    df['date'] = pd.to_datetime(df['date'])
    r = len(df)
    rows += r
    stocks.update(df['code'].astype(str).unique())
    mn, mx = df['date'].min(), df['date'].max()
    global_min = mn if global_min is None or mn < global_min else global_min
    global_max = mx if global_max is None or mx > global_max else global_max
    dup = int(df.duplicated(['code','date']).sum())
    duplicates += dup
    bad = (df['high'] < df[['open','close']].max(axis=1)) | (df['low'] > df[['open','close']].min(axis=1)) | (df[['open','high','low','close']] <= 0).any(axis=1)
    bv = df['volume'].isna() | (df['volume'] < 0)
    bad_ohlc += int(bad.sum())
    bad_volume += int(bv.sum())
    reports.append({'file':f.name,'rows':r,'stocks':int(df['code'].nunique()),'min_date':str(mn.date()),'max_date':str(mx.date()),'duplicates':dup,'bad_ohlc':int(bad.sum()),'bad_volume':int(bv.sum()),'columns':list(df.columns)})

result = {
    'rows': rows,
    'stocks': len(stocks),
    'min_date': str(global_min.date()),
    'max_date': str(global_max.date()),
    'duplicates_code_date': duplicates,
    'bad_ohlc_rows': bad_ohlc,
    'bad_volume_rows': bad_volume,
    'files': reports,
    'notes': [
        'This is a data-integrity audit only; no model fitting yet.',
        'The source is a public historical dataset. We will test survivorship bias and adjusted-price semantics before final backtest.',
        'The first research label will use T+1 open as entry and the next 20 trading sessions for MFE/MAE/T+20.'
    ]
}
(OUT/'data_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
