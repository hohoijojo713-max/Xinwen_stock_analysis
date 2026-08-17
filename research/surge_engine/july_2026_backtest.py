from pathlib import Path
import json
import numpy as np
import pandas as pd

DATA = Path('research/surge_engine/data')
OUT = Path('research/surge_engine/results')
OUT.mkdir(parents=True, exist_ok=True)

# 固化自当前 Step 8 研究结果：
# A = r5 + MA60
# B = MA20斜率 + MA60
# C = MA20位置 + MA60
RULES = {
    'A': {
        'name': 'A：r5 + MA60',
        'ranges': {'r5': (0.0211, 0.0392), 'ma60_gap': (-0.702, -0.128)},
    },
    'B': {
        'name': 'B：MA20斜率 + MA60',
        'ranges': {'ma20_slope': (-0.571, -0.108), 'ma60_gap': (-0.702, -0.128)},
    },
    'C': {
        'name': 'C：MA20位置 + MA60',
        'ranges': {'ma20_gap': (0.0241, 0.0426), 'ma60_gap': (-0.702, -0.128)},
    },
}

SIGNAL_START = pd.Timestamp('2026-07-01')
SIGNAL_END = pd.Timestamp('2026-07-31')
TRAIL_PCT = 0.02
BUY_COMM = 0.0003
SELL_COMM = 0.0003
STAMP = 0.0005
SLIPPAGE = 0.001

parts = []
for f in sorted(DATA.glob('kline_*.parquet')):
    x = pd.read_parquet(f, columns=['code','date','open','high','low','close','volume','amount','pctChg'])
    x['date'] = pd.to_datetime(x['date'])
    x = x.sort_values(['code','date'])
    g = x.groupby('code', group_keys=False)
    ma20 = g.close.transform(lambda s: s.rolling(20).mean())
    ma60 = g.close.transform(lambda s: s.rolling(60).mean())
    x['r5'] = g.close.pct_change(5)
    x['ma20_gap'] = x.close / ma20 - 1
    x['ma60_gap'] = x.close / ma60 - 1
    x['ma20_slope'] = ma20.groupby(x.code).pct_change(20)
    parts.append(x)

df = pd.concat(parts, ignore_index=True).sort_values(['code','date']).reset_index(drop=True)


def net_return(entry_open, exit_price):
    # 按开盘实际成交价格做买入滑点，卖出按退出价再加卖出滑点。
    buy_px = entry_open * (1 + SLIPPAGE)
    sell_px = exit_price * (1 - SLIPPAGE)
    gross = sell_px / buy_px - 1
    # 佣金买卖双边 + 卖出印花税
    net = ((sell_px * (1 - SELL_COMM - STAMP)) / (buy_px * (1 + BUY_COMM))) - 1
    return float(gross), float(net), float(buy_px), float(sell_px)


def simulate(code, dates, trail_pct=TRAIL_PCT):
    gp = df[df.code.eq(code)].reset_index(drop=True)
    idx = {pd.Timestamp(d): i for i, d in enumerate(gp.date)}
    rows = []
    for signal_date in dates:
        i = idx.get(pd.Timestamp(signal_date))
        if i is None or i + 1 >= len(gp):
            continue
        buy_i = i + 1
        entry_raw = float(gp.open.iloc[buy_i])
        entry = entry_raw * (1 + SLIPPAGE)
        stop = entry * 0.95
        peak = entry
        mode = 'hard_stop'
        exit_raw = float(gp.close.iloc[min(buy_i + 19, len(gp)-1)])
        exit_date = pd.Timestamp(gp.date.iloc[min(buy_i + 19, len(gp)-1)])
        reason = 't20'
        max_mfe = -1.0
        min_mae = 1.0
        end = min(buy_i + 19, len(gp)-1)
        for j in range(buy_i, end + 1):
            hi = float(gp.high.iloc[j])
            lo = float(gp.low.iloc[j])
            cl = float(gp.close.iloc[j])
            max_mfe = max(max_mfe, hi / entry - 1)
            min_mae = min(min_mae, lo / entry - 1)
            peak = max(peak, hi)
            # 保守顺序：止损先于止盈升级；同日不假设先冲高再回落。
            if lo <= stop:
                exit_raw = stop
                exit_date = pd.Timestamp(gp.date.iloc[j])
                reason = mode
                break
            peak_mfe = peak / entry - 1
            if peak_mfe >= 0.05:
                new_stop = peak * (1 - trail_pct)
                if new_stop > stop:
                    stop = new_stop
                mode = 'trail'
            elif peak_mfe >= 0.03:
                stop = max(stop, entry * 1.01)
                mode = 'protect_1pct'
            exit_raw = cl
            exit_date = pd.Timestamp(gp.date.iloc[j])
        gross, net, buy_px, sell_px = net_return(entry_raw, exit_raw)
        rows.append({
            'signal_date': str(pd.Timestamp(signal_date).date()),
            'buy_date': str(pd.Timestamp(gp.date.iloc[buy_i]).date()),
            'exit_date': str(exit_date.date()),
            'code': str(code),
            'buy_price_raw': entry_raw,
            'buy_price_effective': buy_px,
            'exit_price_raw': exit_raw,
            'exit_price_effective': sell_px,
            'gross_return_pct': gross * 100,
            'net_return_pct': net * 100,
            'max_mfe_pct': max_mfe * 100,
            'max_mae_pct': min_mae * 100,
            'exit_reason': reason,
        })
    return rows

all_trades = []
rule_summaries = []
for rule_id, spec in RULES.items():
    ranges = spec['ranges']
    for code, gp in df.groupby('code', sort=False):
        g = gp[(gp.date >= SIGNAL_START) & (gp.date <= SIGNAL_END)].copy()
        if g.empty:
            continue
        mask = pd.Series(True, index=g.index)
        for factor, (lo, hi) in ranges.items():
            mask &= g[factor].between(lo, hi, inclusive='both')
        dates = g.loc[mask, 'date'].tolist()
        if not dates:
            continue
        for trade in simulate(code, dates):
            trade['rule'] = rule_id
            trade['rule_name'] = spec['name']
            all_trades.append(trade)

trades = pd.DataFrame(all_trades)
if trades.empty:
    raise SystemExit('No July 2026 signals found.')

trades = trades.sort_values(['rule', 'signal_date', 'code']).reset_index(drop=True)
trades.to_csv(OUT / 'july_2026_trade_detail.csv', index=False, encoding='utf-8-sig')

for rule_id, group in trades.groupby('rule', sort=False):
    reason = group.exit_reason.value_counts(normalize=True).to_dict()
    rule_summaries.append({
        'rule': rule_id,
        'rule_name': RULES[rule_id]['name'],
        'signals_and_trades': int(len(group)),
        'win_rate_gross_pct': float((group.gross_return_pct > 0).mean() * 100),
        'avg_gross_pct': float(group.gross_return_pct.mean()),
        'median_gross_pct': float(group.gross_return_pct.median()),
        'win_rate_net_pct': float((group.net_return_pct > 0).mean() * 100),
        'avg_net_pct': float(group.net_return_pct.mean()),
        'median_net_pct': float(group.net_return_pct.median()),
        'mfe3_pct': float((group.max_mfe_pct >= 3).mean() * 100),
        'mfe5_pct': float((group.max_mfe_pct >= 5).mean() * 100),
        'mfe8_pct': float((group.max_mfe_pct >= 8).mean() * 100),
        'stop5_pct': float((group.exit_reason == 'hard_stop').mean() * 100),
        'protect1_pct': float((group.exit_reason == 'protect_1pct').mean() * 100),
        'trail_pct': float((group.exit_reason == 'trail').mean() * 100),
        't20_pct': float((group.exit_reason == 't20').mean() * 100),
        'net_compounded_independent_trades_pct': float(((1 + group.net_return_pct / 100).prod() - 1) * 100),
        'exit_reason_share': reason,
    })

summary = {
    'signal_window': ['2026-07-01', '2026-07-31'],
    'entry_rule': 'T+1 next trading day open',
    'initial_stop': -5,
    'protect_trigger': 3,
    'protect_stop': 1,
    'trail_trigger': 5,
    'trail_pct': TRAIL_PCT * 100,
    'cost_assumptions': {
        'buy_commission_pct': BUY_COMM * 100,
        'sell_commission_pct': SELL_COMM * 100,
        'stamp_duty_pct': STAMP * 100,
        'slippage_each_side_pct': SLIPPAGE * 100,
    },
    'rules': RULES,
    'summary': rule_summaries,
}
(OUT / 'july_2026_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
