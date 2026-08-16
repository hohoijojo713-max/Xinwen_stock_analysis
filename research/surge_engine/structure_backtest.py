from pathlib import Path
import json
import numpy as np
import pandas as pd

DATA = Path('research/surge_engine/data')
OUT = Path('research/surge_engine/results')
OUT.mkdir(parents=True, exist_ok=True)

# Rebuild the same signal family used by the executable-entry study.
parts = []
for f in sorted(DATA.glob('kline_*.parquet')):
    x = pd.read_parquet(f, columns=['code','date','open','high','low','close','volume','amount','pctChg'])
    x['date'] = pd.to_datetime(x['date'])
    x = x.sort_values(['code','date'])
    g = x.groupby('code', group_keys=False)
    x['r20'] = g.close.pct_change(20)
    ma20 = g.close.transform(lambda s: s.rolling(20).mean())
    ma60 = g.close.transform(lambda s: s.rolling(60).mean())
    x['ma20_gap'] = x.close / ma20 - 1
    x['ma60_gap'] = x.close / ma60 - 1
    x['ma20_slope'] = g.close.transform(lambda s: s.rolling(20).mean()).groupby(x.code).pct_change(20)
    x = x[(x.date >= '2022-01-04') & (x.date <= '2026-03-19')].copy()
    parts.append(x)

df = pd.concat(parts, ignore_index=True).sort_values(['code','date']).reset_index(drop=True)

# Signal family: deliberately the same broad high-surge state used in Step 6A.
signal = (
    df['ma60_gap'].between(-0.604, -0.117) &
    df['ma20_slope'].between(-0.515, -0.0981) &
    df['r20'].between(-0.776, -0.12)
)

# Shift next-session fields onto each signal row.
grp = df.groupby('code', group_keys=False)
for c in ['open','high','low','close','volume','amount','pctChg']:
    df['next_' + c] = grp[c].shift(-1)
df['next_date'] = grp.date.shift(-1)

sig = df.loc[signal & df.next_date.notna()].copy()

# Forward 20-session MFE/MAE from T+1 open, same convention as Step 2.
rows = []
for code, gp in df.groupby('code', sort=False):
    gp = gp.reset_index(drop=True)
    mask = signal.loc[gp.index].to_numpy() if len(signal) else np.zeros(len(gp), dtype=bool)
    valid_idx = np.flatnonzero(mask)
    if len(gp) < 22:
        continue
    for i in valid_idx:
        if i + 20 >= len(gp):
            continue
        s = gp.iloc[i]
        n = gp.iloc[i + 1]
        entry = float(n.open)
        if not np.isfinite(entry) or entry <= 0:
            continue
        future_hi = float(gp.high.iloc[i+1:i+21].max())
        future_lo = float(gp.low.iloc[i+1:i+21].min())
        rows.append({
            'code': code,
            'signal_date': s.date,
            'next_date': n.date,
            'signal_close': float(s.close),
            'next_open': entry,
            'next_high': float(n.high),
            'next_low': float(n.low),
            'next_close': float(n.close),
            'next_gap': entry / float(s.close) - 1,
            'next_open_to_high': float(n.high) / entry - 1,
            'next_open_to_low': float(n.low) / entry - 1,
            'next_open_to_close': float(n.close) / entry - 1,
            'next_close_position': ((float(n.close) - float(n.low)) / (float(n.high) - float(n.low))) if float(n.high) > float(n.low) else 0.5,
            'future_mfe': future_hi / entry - 1,
            'future_mae': future_lo / entry - 1,
        })

res = pd.DataFrame(rows)
if res.empty:
    raise SystemExit('No signal rows available')

# Daily structure classes. These are observable only on T+1, so they are analysis labels, not forecasting inputs.
gap = res.next_gap
open_high = res.next_open_to_high
open_low = res.next_open_to_low
open_close = res.next_open_to_close

res['structure'] = np.select(
    [
        gap >= 0.03,
        (gap < 0.03) & (open_low <= -0.03) & (open_close >= 0.01),
        (gap < 0.03) & (open_low <= -0.03) & (open_close < 0.01),
        (gap < 0.03) & (open_low > -0.03) & (open_high >= 0.03) & (open_close >= 0.01),
        (gap < 0.03) & (open_low > -0.03) & (open_high >= 0.03) & (open_close < 0.01),
    ],
    [
        'open_surge',
        'dip_then_recover',
        'dip_then_fail',
        'intraday_surge_hold',
        'intraday_surge_fade',
    ],
    default='quiet'
)

# Also create cleaner behavioural buckets from the next-day candle itself.
res['next_day_range'] = (res.next_high / res.next_low - 1).replace([np.inf, -np.inf], np.nan)
res['next_day_body'] = res.next_close / res.next_open - 1

summary = {}
for st, gp in res.groupby('structure'):
    summary[st] = {
        'n': int(len(gp)),
        'share': float(len(gp) / len(res)),
        'mfe3_rate': float((gp.future_mfe >= 0.03).mean()),
        'mfe5_rate': float((gp.future_mfe >= 0.05).mean()),
        'mfe8_rate': float((gp.future_mfe >= 0.08).mean()),
        'avg_mfe': float(gp.future_mfe.mean()),
        'median_mfe': float(gp.future_mfe.median()),
        'fail5_rate': float((gp.future_mae <= -0.05).mean()),
        'avg_mae': float(gp.future_mae.mean()),
        'avg_next_day_open_to_close': float(gp.next_open_to_close.mean()),
        'median_next_day_open_to_close': float(gp.next_open_to_close.median()),
    }

# Critical diagnostic: compare signal states that lead to high upside with states that suffer deep adverse excursion.
res['outcome_bucket'] = np.select(
    [
        (res.future_mfe >= 0.03) & (res.future_mae > -0.05),
        (res.future_mfe >= 0.03) & (res.future_mae <= -0.05),
        (res.future_mfe < 0.03) & (res.future_mae > -0.05),
    ],
    ['good_surge', 'volatile_surge', 'quiet'],
    default='bad'
)

bucket_summary = {}
for b, gp in res.groupby('outcome_bucket'):
    bucket_summary[b] = {
        'n': int(len(gp)),
        'share': float(len(gp) / len(res)),
        'avg_mfe': float(gp.future_mfe.mean()),
        'median_mfe': float(gp.future_mfe.median()),
        'avg_mae': float(gp.future_mae.mean()),
        'next_gap_mean': float(gp.next_gap.mean()),
        'next_open_to_high_mean': float(gp.next_open_to_high.mean()),
        'next_open_to_low_mean': float(gp.next_open_to_low.mean()),
        'next_open_to_close_mean': float(gp.next_open_to_close.mean()),
        'share_open_surge': float((gp.structure == 'open_surge').mean()),
        'share_dip_then_recover': float((gp.structure == 'dip_then_recover').mean()),
        'share_intraday_surge_hold': float((gp.structure == 'intraday_surge_hold').mean()),
        'share_intraday_surge_fade': float((gp.structure == 'intraday_surge_fade').mean()),
    }

# Train/validation report for structure effects.
cut = pd.Timestamp('2025-01-01')
res['sample'] = np.where(res.signal_date < cut, 'train', 'valid')
periods = {}
for period, gp in res.groupby('sample'):
    periods[period] = {}
    for st, sg in gp.groupby('structure'):
        periods[period][st] = {
            'n': int(len(sg)),
            'mfe3_rate': float((sg.future_mfe >= 0.03).mean()),
            'mfe5_rate': float((sg.future_mfe >= 0.05).mean()),
            'mfe8_rate': float((sg.future_mfe >= 0.08).mean()),
            'median_mfe': float(sg.future_mfe.median()),
            'fail5_rate': float((sg.future_mae <= -0.05).mean()),
            't20_proxy': float((sg.next_open_to_close).mean()),
        }

out = {
    'objective': 'Separate next-day price paths into open_surge / dip_then_recover / dip_then_fail / intraday_surge_hold / intraday_surge_fade and compare forward MFE/MAE.',
    'signals': int(len(res)),
    'structures': summary,
    'outcome_buckets': bucket_summary,
    'train_valid': periods,
}

(OUT/'structure_backtest_summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
res.to_parquet(OUT/'structure_backtest_signals.parquet', index=False)
print(json.dumps(out, ensure_ascii=False, indent=2))
