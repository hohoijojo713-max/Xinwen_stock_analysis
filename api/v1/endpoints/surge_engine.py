# -*- coding: utf-8 -*-
"""Surge Engine: date-range backtest + current-day screening."""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()
CACHE = Path(os.getenv("SURGE_ENGINE_CACHE", "/tmp/surge_engine_cache"))
CACHE.mkdir(parents=True, exist_ok=True)
BASE = "https://raw.githubusercontent.com/newbiestring-lang/astock/main"
FILES = ["kline_000.parquet", "kline_002.parquet", "kline_300.parquet", "kline_600.parquet", "kline_688.parquet", "kline_other.parquet", "stock_list.parquet"]

STRATEGIES = {
    "A": {"name": "r5 + MA60", "r5": (0.063, 1.492), "ma60_gap": (-0.604, -0.131)},
    "B": {"name": "MA20斜率 + MA60", "ma20_slope": (-0.515, -0.0981), "ma60_gap": (-0.604, -0.131)},
    "C": {"name": "MA20位置 + MA60", "ma20_gap": (-0.529, -0.0688), "ma60_gap": (-0.604, -0.117)},
}

class BacktestRequest(BaseModel):
    start_date: str
    end_date: str
    strategies: list[str] = Field(default_factory=lambda: ["A", "B", "C"])
    commission_bps: float = 3.0
    stamp_tax_bps: float = 10.0
    slippage_bps: float = 5.0
    trail_pct: float = 0.03

class ScreenRequest(BaseModel):
    date: str | None = None
    limit: int = Field(default=30, ge=1, le=100)


def _download() -> list[Path]:
    paths = []
    for name in FILES[:-1]:
        p = CACHE / name
        if not p.exists() or p.stat().st_size < 1024:
            r = requests.get(f"{BASE}/{name}", timeout=120)
            r.raise_for_status()
            p.write_bytes(r.content)
        paths.append(p)
    return paths


def _load() -> pd.DataFrame:
    cached = [CACHE / f for f in FILES[:-1]]
    if not all(p.exists() for p in cached):
        _download()
    parts = []
    for p in cached:
        x = pd.read_parquet(p, columns=["code", "date", "open", "high", "low", "close", "volume", "amount", "pctChg"])
        x["date"] = pd.to_datetime(x["date"])
        parts.append(x)
    df = pd.concat(parts, ignore_index=True).sort_values(["code", "date"]).reset_index(drop=True)
    g = df.groupby("code", group_keys=False)
    ma20 = g.close.transform(lambda s: s.rolling(20).mean())
    ma60 = g.close.transform(lambda s: s.rolling(60).mean())
    df["r5"] = g.close.pct_change(5)
    df["ma20_gap"] = df.close / ma20 - 1
    df["ma60_gap"] = df.close / ma60 - 1
    df["ma20_slope"] = ma20.groupby(df.code).pct_change(20)
    return df


def _match(row: pd.Series, sid: str) -> bool:
    s = STRATEGIES[sid]
    return all(pd.notna(row[f]) and s[f][0] <= float(row[f]) <= s[f][1] for f in s if f != "name")


def _simulate(gp: pd.DataFrame, i: int, fee: float, stamp: float, slip: float, trail_pct: float) -> dict[str, Any] | None:
    if i + 1 >= len(gp):
        return None
    entry_raw = float(gp.open.iloc[i + 1])
    if not math.isfinite(entry_raw) or entry_raw <= 0:
        return None
    entry = entry_raw * (1 + fee + slip)
    stop = entry * 0.95
    peak = entry
    mode = "hard_stop"
    exit_px = float(gp.close.iloc[min(i + 20, len(gp) - 1)])
    reason = "t20"
    end = min(i + 20, len(gp) - 1)
    for j in range(i + 1, end + 1):
        hi, lo, cl = map(float, (gp.high.iloc[j], gp.low.iloc[j], gp.close.iloc[j]))
        peak = max(peak, hi)
        if lo <= stop:
            exit_px, reason = stop, mode
            break
        mfe = peak / entry - 1
        if mfe >= 0.05:
            stop = max(stop, peak * (1 - trail_pct))
            mode = "trail"
        elif mfe >= 0.03:
            stop = max(stop, entry * 1.01)
            mode = "protect_1pct"
        exit_px = cl
    exit_net = exit_px * (1 - fee - stamp - slip)
    ret = exit_net / entry - 1
    return {"entry": entry, "exit": exit_net, "ret": ret, "reason": reason, "mfe": peak / entry - 1}


def _period_rows(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, sid: str, fee: float, stamp: float, slip: float, trail_pct: float):
    out = []
    for code, gp in df.groupby("code", sort=False):
        gp = gp.reset_index(drop=True)
        dates = gp.date.values
        for i in range(60, len(gp) - 1):
            d = pd.Timestamp(dates[i])
            if d < start or d > end:
                continue
            row = gp.iloc[i]
            if not _match(row, sid):
                continue
            trade = _simulate(gp, i, fee, stamp, slip, trail_pct)
            if trade is None:
                continue
            out.append({"code": str(code), "signal_date": d.strftime("%Y-%m-%d"), "entry_date": pd.Timestamp(dates[i+1]).strftime("%Y-%m-%d"), **trade})
    return out

@router.post("/backtest")
def backtest(req: BacktestRequest):
    try:
        start, end = pd.Timestamp(req.start_date), pd.Timestamp(req.end_date)
        if start > end:
            raise ValueError("开始日期不能晚于结束日期")
        df = _load()
        fee, stamp, slip = req.commission_bps / 10000, req.stamp_tax_bps / 10000, req.slippage_bps / 10000
        selected = [s for s in req.strategies if s in STRATEGIES]
        if not selected:
            raise ValueError("至少选择一套策略")
        results = []
        trades = {}
        for sid in selected:
            rows = _period_rows(df, start, end, sid, fee, stamp, slip, req.trail_pct)
            trades[sid] = rows
            rs = pd.DataFrame(rows)
            if rs.empty:
                metrics = {"strategy": sid, "name": STRATEGIES[sid]["name"], "trades": 0, "win_rate": 0, "avg_return": 0, "net_avg_return": 0, "stop5_rate": 0, "protect_rate": 0, "trail_rate": 0, "avg_mfe": 0, "cum_return": 0, "max_drawdown": 0}
            else:
                rets = rs.ret.astype(float)
                eq = (1 + rets).cumprod()
                peak = eq.cummax()
                dd = eq / peak - 1
                years = max((end - start).days / 365.25, 1 / 365.25)
                metrics = {"strategy": sid, "name": STRATEGIES[sid]["name"], "trades": int(len(rs)), "annual_trades": round(len(rs) / years, 1), "win_rate": round(float((rets > 0).mean()) * 100, 2), "avg_return": round(float(rets.mean()) * 100, 2), "net_avg_return": round(float(rets.mean()) * 100, 2), "stop5_rate": round(float((rs.reason == "hard_stop").mean()) * 100, 2), "protect_rate": round(float((rs.reason == "protect_1pct").mean()) * 100, 2), "trail_rate": round(float((rs.reason == "trail").mean()) * 100, 2), "avg_mfe": round(float(rs.mfe.mean()) * 100, 2), "cum_return": round(float(eq.iloc[-1] - 1) * 100, 2), "max_drawdown": round(float(dd.min()) * 100, 2)}
            results.append(metrics)
        return {"start_date": req.start_date, "end_date": req.end_date, "results": results, "trades": trades}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"冲高引擎回测失败: {exc}") from exc

@router.post("/screen")
def screen(req: ScreenRequest):
    try:
        df = _load()
        d = pd.Timestamp(req.date) if req.date else df.date.max()
        day = df[df.date.eq(d)].copy()
        picks = []
        for _, row in day.iterrows():
            hits = [sid for sid in STRATEGIES if _match(row, sid)]
            if hits:
                score = len(hits) * 25 + min(max(float(row.get("r5", 0) or 0) * 100, -20), 30)
                picks.append({"code": str(row.code), "date": d.strftime("%Y-%m-%d"), "strategies": hits, "score": round(score, 2), "close": float(row.close), "r5": float(row.r5) if pd.notna(row.r5) else None, "ma20_gap": float(row.ma20_gap) if pd.notna(row.ma20_gap) else None, "ma60_gap": float(row.ma60_gap) if pd.notna(row.ma60_gap) else None})
        picks.sort(key=lambda x: x["score"], reverse=True)
        return {"date": d.strftime("%Y-%m-%d"), "count": len(picks), "items": picks[: req.limit]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"今日选股失败: {exc}") from exc
