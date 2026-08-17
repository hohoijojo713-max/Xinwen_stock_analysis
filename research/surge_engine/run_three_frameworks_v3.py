from __future__ import annotations

import concurrent.futures
import random
import time
from typing import List

import pandas as pd

from august_2026_three_framework_backtest_v2 import main as original_main
import august_2026_three_framework_backtest_v2 as m

# Eastmoney may rate-limit the one-shot universe endpoint. The research parquet
# universe is the authoritative universe for this historical backtest, so do not
# make the whole run depend on a live universe request.
def safe_universe():
    base = m.load_base()
    return set(base['code'].astype(str).str.zfill(6).unique())


def safe_fetch_5m(codes: List[str]) -> pd.DataFrame:
    rows = []
    # Lower concurrency materially reduces Eastmoney 429 / disconnect errors.
    workers = 8
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(m.fetch_one_5m, c) for c in codes]
        for n, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            code, klines = fut.result()
            if n % 250 == 0:
                print(f'5m fetched {n}/{len(codes)}', flush=True)
            for p in klines:
                if len(p) >= 6:
                    rows.append({
                        'code': code,
                        'datetime': p[0],
                        'open': p[1],
                        'close': p[2],
                        'high': p[3],
                        'low': p[4],
                        'volume': p[5],
                    })
            if n % 100 == 0:
                time.sleep(0.2 + random.random() * 0.3)
    if not rows:
        raise RuntimeError('No 5m data returned from Eastmoney')
    x = pd.DataFrame(rows, columns=['code','datetime','open','close','high','low','volume'])
    for c in ['open','close','high','low','volume']:
        x[c] = pd.to_numeric(x[c], errors='coerce')
    x['datetime'] = pd.to_datetime(x['datetime'])
    x['code'] = x['code'].astype(str).str.zfill(6)
    return x.sort_values(['code','datetime']).reset_index(drop=True)

m.get_universe = safe_universe
m.fetch_5m = safe_fetch_5m

if __name__ == '__main__':
    original_main()
