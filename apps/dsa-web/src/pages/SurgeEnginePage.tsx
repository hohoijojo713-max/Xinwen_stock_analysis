import type React from 'react';
import { useMemo, useState } from 'react';
import { BarChart3, CalendarDays, Play, RefreshCw, ShieldCheck, Sparkles, TrendingUp } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Card, Badge } from '../components/common';
import { surgeEngineApi, type SurgeBacktestResponse, type SurgeStrategyId } from '../api/surgeEngine';

const STRATEGIES: { id: SurgeStrategyId; name: string; desc: string }[] = [
  { id: 'A', name: '策略 A · r5 + MA60', desc: '偏收益平衡：5日动量 + 60日位置' },
  { id: 'B', name: '策略 B · MA20斜率 + MA60', desc: '偏低止损：20日均线斜率 + 60日位置' },
  { id: 'C', name: '策略 C · MA20位置 + MA60', desc: '偏简单高频：20日位置 + 60日位置' },
];

const pct = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;

const SurgeEnginePage: React.FC = () => {
  const [start, setStart] = useState('2024-01-01');
  const [end, setEnd] = useState('2025-12-31');
  const [selected, setSelected] = useState<SurgeStrategyId[]>(['A', 'B', 'C']);
  const [commission, setCommission] = useState('3');
  const [stamp, setStamp] = useState('10');
  const [slippage, setSlippage] = useState('5');
  const [trail, setTrail] = useState('3');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SurgeBacktestResponse | null>(null);
  const [error, setError] = useState('');

  const chartData = useMemo(
    () => (result?.results ?? []).map((r) => ({ name: `策略 ${r.strategy}`, 胜率: r.win_rate, 平均收益: r.avg_return, 止损率: r.stop5_rate })),
    [result],
  );

  const toggle = (id: SurgeStrategyId) => setSelected((cur) => cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id]);

  const run = async () => {
    setError('');
    if (!selected.length) { setError('至少选择一套策略'); return; }
    if (start > end) { setError('开始日期不能晚于结束日期'); return; }
    setLoading(true);
    try {
      const data = await surgeEngineApi.backtest({
        start_date: start, end_date: end, strategies: selected,
        commission_bps: Number(commission) || 0,
        stamp_tax_bps: Number(stamp) || 0,
        slippage_bps: Number(slippage) || 0,
        trail_pct: (Number(trail) || 3) / 100,
      });
      setResult(data);
    } catch (e) {
      console.error(e);
      setError('回测失败。第一次运行可能需要下载研究数据，请确认后端已安装 pyarrow 且网络可用。');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto flex w-full max-w-[1500px] flex-col gap-6 p-6 lg:p-8">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-muted-text"><Sparkles className="h-4 w-4" /> SURGE ENGINE</div>
          <h1 className="text-3xl font-semibold text-primary-text">冲高选股引擎</h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-secondary-text">指定任意历史时间段，直接比较三套候选规则。回测采用 T+1 开盘入场、-5% 初始止损、+3% 保护到 +1%、+5% 后跟踪止盈。</p>
        </div>
        <Badge variant="success">研究版 V1</Badge>
      </div>

      <Card padding="lg">
        <div className="grid gap-5 lg:grid-cols-4">
          <label className="space-y-2 text-sm"><span className="text-secondary-text">开始日期</span><div className="relative"><CalendarDays className="pointer-events-none absolute left-3 top-3.5 h-4 w-4 text-muted-text"/><input className="input-surface h-11 w-full rounded-xl border bg-transparent pl-10 pr-3" type="date" value={start} onChange={(e) => setStart(e.target.value)} /></div></label>
          <label className="space-y-2 text-sm"><span className="text-secondary-text">结束日期</span><div className="relative"><CalendarDays className="pointer-events-none absolute left-3 top-3.5 h-4 w-4 text-muted-text"/><input className="input-surface h-11 w-full rounded-xl border bg-transparent pl-10 pr-3" type="date" value={end} onChange={(e) => setEnd(e.target.value)} /></div></label>
          <label className="space-y-2 text-sm"><span className="text-secondary-text">跟踪止盈（%）</span><input className="input-surface h-11 w-full rounded-xl border bg-transparent px-3" value={trail} onChange={(e) => setTrail(e.target.value)} inputMode="decimal" /></label>
          <div className="flex items-end"><button className="btn-primary h-11 w-full" type="button" onClick={() => void run()} disabled={loading}>{loading ? <RefreshCw className="mx-auto h-4 w-4 animate-spin" /> : <><Play className="mr-2 h-4 w-4" />开始回测</>}</button></div>
        </div>
        <div className="mt-5 grid gap-3 md:grid-cols-3">
          {STRATEGIES.map((s) => <button key={s.id} type="button" onClick={() => toggle(s.id)} className={`rounded-2xl border p-4 text-left transition ${selected.includes(s.id) ? 'border-accent bg-accent/10' : 'border-white/10 bg-white/[0.02]'}`}><div className="flex items-center justify-between"><span className="font-medium text-primary-text">{s.name}</span><span className="text-xs text-secondary-text">{selected.includes(s.id) ? '已选' : '未选'}</span></div><p className="mt-2 text-xs leading-5 text-muted-text">{s.desc}</p></button>)}
        </div>
        <div className="mt-5 grid gap-3 md:grid-cols-3"><label className="space-y-1 text-xs text-muted-text">佣金（bps）<input className="input-surface mt-1 h-10 w-full rounded-xl border bg-transparent px-3" value={commission} onChange={(e) => setCommission(e.target.value)} /></label><label className="space-y-1 text-xs text-muted-text">印花税（bps）<input className="input-surface mt-1 h-10 w-full rounded-xl border bg-transparent px-3" value={stamp} onChange={(e) => setStamp(e.target.value)} /></label><label className="space-y-1 text-xs text-muted-text">滑点（bps）<input className="input-surface mt-1 h-10 w-full rounded-xl border bg-transparent px-3" value={slippage} onChange={(e) => setSlippage(e.target.value)} /></label></div>
        {error && <div className="mt-4 rounded-xl border border-red-400/20 bg-red-400/10 p-3 text-sm text-red-200">{error}</div>}
      </Card>

      {result && <>
        <div className="grid gap-4 lg:grid-cols-3">
          {result.results.map((r) => <Card key={r.strategy} padding="md"><div className="flex items-center justify-between"><div><div className="text-xs uppercase tracking-wider text-muted-text">策略 {r.strategy}</div><div className="mt-1 text-lg font-semibold text-primary-text">{r.name}</div></div><ShieldCheck className="h-5 w-5 text-success" /></div><div className="mt-5 grid grid-cols-2 gap-3 text-sm"><div><div className="text-muted-text">交易次数</div><div className="mt-1 font-semibold text-primary-text">{r.trades}</div></div><div><div className="text-muted-text">年化交易</div><div className="mt-1 font-semibold text-primary-text">{r.annual_trades ?? '--'}</div></div><div><div className="text-muted-text">胜率</div><div className="mt-1 font-semibold text-success">{r.win_rate.toFixed(2)}%</div></div><div><div className="text-muted-text">净收益/笔</div><div className="mt-1 font-semibold text-primary-text">{pct(r.net_avg_return)}</div></div><div><div className="text-muted-text">-5%止损</div><div className="mt-1 font-semibold text-warning">{r.stop5_rate.toFixed(2)}%</div></div><div><div className="text-muted-text">最大回撤</div><div className="mt-1 font-semibold text-danger">{pct(r.max_drawdown)}</div></div></div></Card>)}
        </div>
        <Card padding="lg"><div className="mb-4 flex items-center gap-2"><BarChart3 className="h-4 w-4 text-accent"/><span className="font-medium text-primary-text">策略横向比较</span></div><div className="h-[320px]"><ResponsiveContainer width="100%" height="100%"><BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}><CartesianGrid strokeDasharray="3 3" opacity={0.12}/><XAxis dataKey="name"/><YAxis unit="%"/><Tooltip/><Bar dataKey="胜率" fill="var(--accent)" radius={[6,6,0,0]}/><Bar dataKey="平均收益" fill="var(--success)" radius={[6,6,0,0]}/><Bar dataKey="止损率" fill="var(--warning)" radius={[6,6,0,0]}/></BarChart></ResponsiveContainer></div></Card>
        <Card padding="lg"><div className="mb-4 flex items-center gap-2"><TrendingUp className="h-4 w-4 text-accent"/><span className="font-medium text-primary-text">逐笔交易明细</span></div><div className="overflow-auto"><table className="min-w-full text-sm"><thead><tr className="border-b border-white/10 text-left text-xs text-muted-text"><th className="px-3 py-2">策略</th><th className="px-3 py-2">股票</th><th className="px-3 py-2">信号日</th><th className="px-3 py-2">买入日</th><th className="px-3 py-2">入场</th><th className="px-3 py-2">收益</th><th className="px-3 py-2">退出</th></tr></thead><tbody>{selected.flatMap((sid) => (result.trades[sid] ?? []).slice(0, 100).map((t, idx) => <tr key={`${sid}-${t.code}-${t.signal_date}-${idx}`} className="border-b border-white/5"><td className="px-3 py-2">策略 {sid}</td><td className="px-3 py-2 font-mono">{t.code}</td><td className="px-3 py-2">{t.signal_date}</td><td className="px-3 py-2">{t.entry_date}</td><td className="px-3 py-2">{t.entry.toFixed(2)}</td><td className={`px-3 py-2 font-medium ${t.ret >= 0 ? 'text-success' : 'text-danger'}`}>{pct(t.ret * 100)}</td><td className="px-3 py-2">{t.reason}</td></tr>))}</tbody></table></div></Card>
      </>}
    </div>
  );
};

export default SurgeEnginePage;
