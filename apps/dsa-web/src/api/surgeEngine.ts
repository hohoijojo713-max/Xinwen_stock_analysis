import apiClient from './index';

export type SurgeStrategyId = 'A' | 'B' | 'C';
export type SurgeMetric = {
  strategy: SurgeStrategyId;
  name: string;
  trades: number;
  annual_trades?: number;
  win_rate: number;
  avg_return: number;
  net_avg_return: number;
  stop5_rate: number;
  protect_rate: number;
  trail_rate: number;
  avg_mfe: number;
  cum_return: number;
  max_drawdown: number;
};

export type SurgeTrade = {
  code: string;
  signal_date: string;
  entry_date: string;
  entry: number;
  exit: number;
  ret: number;
  reason: string;
  mfe: number;
};

export type SurgeBacktestResponse = {
  start_date: string;
  end_date: string;
  results: SurgeMetric[];
  trades: Record<string, SurgeTrade[]>;
};

export type SurgeScreenItem = {
  code: string;
  date: string;
  strategies: SurgeStrategyId[];
  score: number;
  close: number;
  r5: number | null;
  ma20_gap: number | null;
  ma60_gap: number | null;
};

export type SurgeScreenResponse = { date: string; count: number; items: SurgeScreenItem[] };

export const surgeEngineApi = {
  backtest: async (payload: {
    start_date: string;
    end_date: string;
    strategies: SurgeStrategyId[];
    commission_bps: number;
    stamp_tax_bps: number;
    slippage_bps: number;
    trail_pct: number;
  }) => (await apiClient.post<SurgeBacktestResponse>('/api/v1/surge-engine/backtest', payload)).data,
  screen: async (date?: string, limit = 30) => (await apiClient.post<SurgeScreenResponse>('/api/v1/surge-engine/screen', { date, limit })).data,
};
