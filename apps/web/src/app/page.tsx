"use client";

import {
  Activity,
  ArrowUpRight,
  Bot,
  Boxes,
  BrainCircuit,
  CircleDollarSign,
  Clock3,
  Database,
  FlaskConical,
  GitCompareArrows,
  Play,
  Radio,
  Send,
  ShieldCheck,
  Sparkles,
  TerminalSquare,
  X,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

import {
  type AccountTrace,
  type Experiment,
  type ProductContext,
  type Recommendation,
  type RunEvent,
  createChatRun,
  experimentStreamUrl,
  readAccount,
  readContext,
  readExperiment,
  runEventSchema,
} from "@/lib/api";

const DEFAULT_COMMAND =
  "Run the Sidekick manufacturing ICP as of 2026-06-30. Compare heuristic, LightGBM, and TabPFN. Return top 25.";

const EVENT_LABELS: Record<string, string> = {
  "run.queued": "Experiment admitted",
  "run.started": "Execution started",
  "data.loaded": "Frozen corpus loaded",
  "factsheets.materialized": "FactSheets materialized",
  "accounts.consolidated": "Account state consolidated",
  "training_set.built": "Point-in-time training set built",
  "portfolio.proposed": "Signal portfolio weights proposed",
  "models.benchmarked": "Models benchmarked",
  "model.artifact_written": "Versioned model artifact written",
  "model.unavailable": "Model unavailable",
  "recommendations.created": "Decision policy completed",
  "run.completed": "Run completed",
  "run.failed": "Run failed",
};

function pct(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function money(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

function actionLabel(action: Recommendation["action"]): string {
  return action === "review" ? "human review" : action.replaceAll("_", " ");
}

function compactPayload(payload: Record<string, unknown>): string {
  return Object.entries(payload)
    .filter(([, value]) => typeof value !== "object")
    .slice(0, 4)
    .map(([key, value]) => `${key}: ${String(value)}`)
    .join(" · ");
}

export default function Home() {
  const [context, setContext] = useState<ProductContext | null>(null);
  const [command, setCommand] = useState(DEFAULT_COMMAND);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [selectedAccount, setSelectedAccount] = useState<Recommendation | null>(
    null,
  );
  const [accountTrace, setAccountTrace] = useState<AccountTrace | null>(null);
  const accountTraceCache = useRef(new Map<string, AccountTrace>());
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function loadContext() {
      for (let attempt = 0; attempt < 8 && !cancelled; attempt += 1) {
        try {
          const loaded = await readContext();
          if (!cancelled) {
            setContext(loaded);
            setError(null);
          }
          return;
        } catch (cause) {
          if (attempt === 7 && !cancelled) {
            setError(
              cause instanceof Error ? cause.message : "API unavailable",
            );
            return;
          }
          await new Promise((resolve) => setTimeout(resolve, 750));
        }
      }
    }
    void loadContext();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!activeRunId) return;
    const stream = new EventSource(experimentStreamUrl(activeRunId));
    let closedIntentionally = false;
    const receive = (raw: MessageEvent<string>) => {
      let decoded: unknown;
      try {
        decoded = JSON.parse(raw.data) as unknown;
      } catch {
        setError("The engine returned malformed run-event JSON.");
        closedIntentionally = true;
        stream.close();
        return;
      }
      const parsed = runEventSchema.safeParse(decoded);
      if (!parsed.success) {
        setError("The engine returned an invalid run event.");
        closedIntentionally = true;
        stream.close();
        return;
      }
      setEvents((current) =>
        current.some((event) => event.seq === parsed.data.seq)
          ? current
          : [...current, parsed.data].sort(
              (left, right) => left.seq - right.seq,
            ),
      );
      if (
        ["run.completed", "run.failed", "run.cancelled"].includes(
          parsed.data.kind,
        )
      ) {
        closedIntentionally = true;
        stream.close();
        readExperiment(activeRunId)
          .then((result) => {
            setExperiment(result);
            setSelectedAccount(result.recommendations[0] ?? null);
          })
          .catch((cause: unknown) =>
            setError(
              cause instanceof Error ? cause.message : "Unable to read run",
            ),
          )
          .finally(() => setSubmitting(false));
      }
    };
    stream.addEventListener("run-event", receive as EventListener);
    stream.onerror = () => {
      if (!closedIntentionally) setError("Run stream reconnecting…");
    };
    return () => {
      closedIntentionally = true;
      stream.close();
    };
  }, [activeRunId]);

  useEffect(() => {
    if (!selectedAccount) {
      setAccountTrace(null);
      return;
    }
    let cancelled = false;
    const cached = accountTraceCache.current.get(selectedAccount.account_id);
    setAccountTrace(cached ?? null);
    if (cached) return;
    readAccount(selectedAccount.account_id)
      .then((trace) => {
        accountTraceCache.current.set(selectedAccount.account_id, trace);
        if (!cancelled) setAccountTrace(trace);
      })
      .catch(() => {
        if (!cancelled) setAccountTrace(null);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedAccount]);

  useEffect(() => {
    if (!selectedAccount || !experiment) return;
    const index = experiment.recommendations.findIndex(
      (recommendation) =>
        recommendation.account_id === selectedAccount.account_id,
    );
    for (const neighbor of [
      experiment.recommendations[index - 1],
      experiment.recommendations[index + 1],
    ]) {
      if (!neighbor || accountTraceCache.current.has(neighbor.account_id))
        continue;
      void readAccount(neighbor.account_id)
        .then((trace) => {
          accountTraceCache.current.set(neighbor.account_id, trace);
        })
        .catch(() => undefined);
    }
  }, [experiment, selectedAccount]);

  useEffect(() => {
    if (!selectedAccount || !window.matchMedia("(max-width: 1200px)").matches)
      return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, [selectedAccount]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    setEvents([]);
    setExperiment(null);
    setSelectedAccount(null);
    setAccountTrace(null);
    accountTraceCache.current.clear();
    try {
      const result = await createChatRun(command);
      setActiveRunId(result.run.run_id);
    } catch (cause) {
      setSubmitting(false);
      setError(cause instanceof Error ? cause.message : "Run could not start");
    }
  }

  const latestEvent = events.at(-1);
  const llmCalls = useMemo(
    () =>
      events
        .filter((event) => event.kind === "recommendations.created")
        .reduce(
          (total, event) => total + Number(event.payload.llmCalls ?? 0),
          0,
        ),
    [events],
  );
  const proxyBaseRate = useMemo(() => {
    const trainingEvent = events.find(
      (event) => event.kind === "training_set.built",
    );
    const rows = Number(trainingEvent?.payload.rows ?? 0);
    const positives = Number(trainingEvent?.payload.positiveRows ?? 0);
    return rows > 0 ? positives / rows : null;
  }, [events]);
  const recommendations = experiment?.recommendations ?? [];
  const selectedIndex = selectedAccount
    ? recommendations.findIndex(
        (recommendation) =>
          recommendation.account_id === selectedAccount.account_id,
      )
    : -1;

  function moveAccount(offset: number) {
    if (selectedIndex < 0 || recommendations.length === 0) return;
    const nextIndex = Math.min(
      Math.max(selectedIndex + offset, 0),
      recommendations.length - 1,
    );
    setSelectedAccount(recommendations[nextIndex]);
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark">
            <Activity size={19} strokeWidth={2.4} />
          </div>
          <div>
            <div className="eyebrow">CLEAN R&amp;D / LAB 01</div>
            <h1>Signal Simulation Engine</h1>
          </div>
        </div>
        <div className="status-rail">
          <span className="status-chip">
            <Radio size={13} />
            {context ? "ENGINE ONLINE" : "CONNECTING"}
          </span>
          <span className="status-chip muted">
            CORPUS {context?.accounts ?? "—"} ACCOUNTS
          </span>
          <span className="status-chip muted">LLM SPEND LOCKED</span>
        </div>
      </header>

      <section className="workspace">
        <aside className="left-rail">
          <div className="panel-heading">
            <div>
              <span className="section-index">01</span>
              <h2>Experiment chat</h2>
            </div>
            <Bot size={17} />
          </div>

          <div className="chat-thread">
            <div className="message system-message">
              <span className="message-label">HARNESS</span>
              <p>
                Describe the ICP, date, models, and capacity. I will resolve it
                into a frozen experiment specification before execution.
              </p>
            </div>
            {activeRunId && (
              <div className="message user-message">
                <span className="message-label">YOU</span>
                <p>{command}</p>
              </div>
            )}
            {latestEvent && (
              <div className="message system-message live-message">
                <span className="message-label">ENGINE</span>
                <p>{EVENT_LABELS[latestEvent.kind] ?? latestEvent.kind}</p>
                <small>{compactPayload(latestEvent.payload)}</small>
              </div>
            )}
          </div>

          <form className="command-form" onSubmit={submit}>
            <textarea
              value={command}
              onChange={(event) => setCommand(event.target.value)}
              aria-label="Experiment command"
              rows={6}
            />
            <div className="command-actions">
              <span>Typed spec · reproducible</span>
              <button disabled={submitting} type="submit">
                {submitting ? (
                  <Activity className="spin" size={15} />
                ) : (
                  <Send size={15} />
                )}
                {submitting ? "RUNNING" : "RUN"}
              </button>
            </div>
          </form>

          <div className="context-block">
            <div className="mini-label">ACTIVE PRODUCT</div>
            <strong>{context?.product.name ?? "Sidekick"}</strong>
            <p>{context?.product.description ?? "Loading product context…"}</p>
            <a
              href={context?.product.website ?? "https://textside.com/"}
              target="_blank"
              rel="noreferrer"
            >
              textside.com <ArrowUpRight size={12} />
            </a>
          </div>

          <div className="context-block">
            <div className="mini-label">FROZEN ICP</div>
            <strong>{context?.icp.name ?? "US frontline manufacturers"}</strong>
            <p>{context?.goal.objective}</p>
          </div>
        </aside>

        <section className="center-stage">
          <div className="stage-header">
            <div>
              <span className="section-index">02</span>
              <h2>Decision trace</h2>
              <p>
                Structured evidence and execution—not hidden chain-of-thought.
              </p>
            </div>
            <div className="run-identity">
              <span>{activeRunId ? activeRunId.slice(0, 8) : "NO RUN"}</span>
              <b>
                {experiment?.run.state ?? (submitting ? "running" : "idle")}
              </b>
            </div>
          </div>

          <div className="metric-strip">
            <Metric
              icon={<Database size={16} />}
              label="EVENTS"
              value={String(context?.events ?? 0)}
            />
            <Metric
              icon={<Boxes size={16} />}
              label="SIGNAL TYPES"
              value={String(context?.signals.length ?? 0)}
            />
            <Metric
              icon={<CircleDollarSign size={16} />}
              label="LLM CALLS"
              value={String(llmCalls)}
            />
            <Metric
              icon={<ShieldCheck size={16} />}
              label="LABEL"
              value="R&D PROXY"
            />
          </div>

          <div className="trace-and-results">
            <section className="trace-panel">
              <div className="subheading">
                <TerminalSquare size={15} />
                EXECUTION LEDGER
              </div>
              {events.length === 0 ? (
                <EmptyTrace />
              ) : (
                <ol className="trace-list">
                  {events.map((event) => (
                    <li key={event.seq}>
                      <div className="trace-seq">
                        {String(event.seq).padStart(2, "0")}
                      </div>
                      <div className="trace-node" />
                      <div className="trace-copy">
                        <strong>
                          {EVENT_LABELS[event.kind] ?? event.kind}
                        </strong>
                        <p>
                          {compactPayload(event.payload) ||
                            "Structured payload recorded"}
                        </p>
                      </div>
                      <time>
                        {new Date(event.created_at).toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                          second: "2-digit",
                        })}
                      </time>
                    </li>
                  ))}
                </ol>
              )}
            </section>

            <section className="results-panel">
              <div className="subheading">
                <Sparkles size={15} />
                RECOMMENDATIONS
                <span>{experiment?.recommendations.length ?? 0}</span>
              </div>
              <div className="account-list">
                {experiment?.recommendations.map((recommendation) => (
                  <button
                    className={
                      selectedAccount?.account_id === recommendation.account_id
                        ? "account-row selected"
                        : "account-row"
                    }
                    key={recommendation.account_id}
                    onClick={() => setSelectedAccount(recommendation)}
                  >
                    <span className="rank">
                      {String(recommendation.rank).padStart(2, "0")}
                    </span>
                    <span className="account-copy">
                      <strong>
                        {recommendation.account_name ??
                          recommendation.account_id}
                      </strong>
                      <small>
                        {[recommendation.state, recommendation.naics_code]
                          .filter(Boolean)
                          .join(" · ")}{" "}
                        · {actionLabel(recommendation.action)}
                      </small>
                    </span>
                    <span className="score">
                      {(recommendation.signal_relevance_score * 100).toFixed(0)}
                    </span>
                  </button>
                )) ?? <EmptyResults />}
              </div>
            </section>
          </div>

          <section className="benchmark-panel">
            <div className="subheading">
              <GitCompareArrows size={15} />
              MODEL BENCHMARK
              <span>
                POINT-IN-TIME / ACCOUNT-HOLDOUT
                {proxyBaseRate !== null
                  ? ` · BASE RATE ${pct(proxyBaseRate)}`
                  : ""}
              </span>
            </div>
            <div className="benchmark-grid">
              {(experiment?.benchmarks ?? []).map((result) => (
                <article className="benchmark-card" key={result.model_id}>
                  <div className="benchmark-title">
                    <strong>{result.model_id}</strong>
                    <span className={`model-state ${result.status}`}>
                      {result.status}
                    </span>
                  </div>
                  <div className="benchmark-values">
                    <div>
                      <b>{pct(result.pr_auc)}</b>
                      <span>PR-AUC</span>
                    </div>
                    <div>
                      <b>{pct(result.precision_at_k)}</b>
                      <span>PRECISION@K</span>
                    </div>
                    <div>
                      <b>{result.fit_seconds.toFixed(2)}s</b>
                      <span>FIT</span>
                    </div>
                  </div>
                  {proxyBaseRate !== null && result.pr_auc !== null && (
                    <p>
                      {(result.pr_auc / proxyBaseRate).toFixed(1)}× proxy
                      baseline
                    </p>
                  )}
                  {result.detail && <p>{result.detail}</p>}
                </article>
              ))}
              {!experiment && (
                <>
                  <BenchmarkGhost name="heuristic-v1" />
                  <BenchmarkGhost name="lightgbm-v1" />
                  <BenchmarkGhost name="tabpfn-local-v2" />
                </>
              )}
            </div>
          </section>
        </section>

        <aside
          className={
            selectedAccount ? "right-rail has-selection" : "right-rail"
          }
        >
          <div className="panel-heading">
            <div>
              <span className="section-index">03</span>
              <h2>Account inspector</h2>
            </div>
            <div className="inspector-tools">
              <BrainCircuit size={17} />
              {selectedAccount && (
                <button
                  aria-label="Close account inspector"
                  className="inspector-close"
                  onClick={() => setSelectedAccount(null)}
                  type="button"
                >
                  <X size={15} />
                </button>
              )}
            </div>
          </div>
          {selectedAccount ? (
            <>
              <div className="account-switcher">
                <button
                  aria-label="Previous account"
                  disabled={selectedIndex <= 0}
                  onClick={() => moveAccount(-1)}
                  type="button"
                >
                  PREV
                </button>
                <span>
                  ACCOUNT {String(selectedIndex + 1).padStart(2, "0")} /{" "}
                  {String(recommendations.length).padStart(2, "0")}
                </span>
                <button
                  aria-label="Next account"
                  disabled={
                    selectedIndex < 0 ||
                    selectedIndex >= recommendations.length - 1
                  }
                  onClick={() => moveAccount(1)}
                  type="button"
                >
                  NEXT
                </button>
              </div>
              <AccountInspector
                recommendation={selectedAccount}
                trace={accountTrace}
              />
            </>
          ) : (
            <div className="inspector-empty">
              <FlaskConical size={30} />
              <strong>No account selected</strong>
              <p>Run an experiment, then select a ranked manufacturer.</p>
            </div>
          )}
        </aside>
      </section>

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button onClick={() => setError(null)}>DISMISS</button>
        </div>
      )}
    </main>
  );
}

function Metric({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="metric">
      <span>{icon}</span>
      <div>
        <b>{value}</b>
        <small>{label}</small>
      </div>
    </div>
  );
}

function EmptyTrace() {
  return (
    <div className="empty-state">
      <Play size={24} />
      <strong>Awaiting experiment</strong>
      <p>
        Run the default Sidekick scenario to populate the durable execution
        ledger.
      </p>
    </div>
  );
}

function EmptyResults() {
  return (
    <div className="empty-state compact">
      <Clock3 size={20} />
      <strong>No recommendations yet</strong>
    </div>
  );
}

function BenchmarkGhost({ name }: { name: string }) {
  return (
    <article className="benchmark-card ghost">
      <div className="benchmark-title">
        <strong>{name}</strong>
        <span className="model-state">waiting</span>
      </div>
      <div className="benchmark-values">
        <div>
          <b>—</b>
          <span>PR-AUC</span>
        </div>
        <div>
          <b>—</b>
          <span>PRECISION@K</span>
        </div>
        <div>
          <b>—</b>
          <span>FIT</span>
        </div>
      </div>
    </article>
  );
}

function ScoreRow({ label, value }: { label: string; value: number | null }) {
  return (
    <div className="score-breakdown-row">
      <span>{label}</span>
      <div className="score-track">
        <i style={{ width: `${(value ?? 0) * 100}%` }} />
      </div>
      <b>{value === null ? "—" : Math.round(value * 100)}</b>
    </div>
  );
}

function AccountInspector({
  recommendation,
  trace,
}: {
  recommendation: Recommendation;
  trace: AccountTrace | null;
}) {
  const best =
    recommendation.action_evaluations.length > 0
      ? recommendation.action_evaluations.reduce((left, right) =>
          right.expected_value > left.expected_value ? right : left,
        )
      : null;
  const rawTrace = trace;
  return (
    <div className="inspector-content">
      <div className="account-hero">
        <span>RANK {String(recommendation.rank).padStart(2, "0")}</span>
        <h3>
          {rawTrace?.account?.name ??
            recommendation.account_name ??
            "Loading account details…"}
        </h3>
        <p>
          NAICS{" "}
          {rawTrace?.account?.naics_code ?? recommendation.naics_code ?? "—"} ·{" "}
          {rawTrace?.account?.state ?? recommendation.state ?? "—"}
        </p>
        <div className="hero-score">
          <b>{(recommendation.signal_relevance_score * 100).toFixed(0)}</b>
          <span>
            SIGNAL
            <br />
            RELEVANCE
          </span>
        </div>
      </div>

      <div className="decision-box">
        <span>
          {recommendation.action === "review"
            ? "DECISION STATUS"
            : "RECOMMENDED ACTION"}
        </span>
        <strong>{actionLabel(recommendation.action)}</strong>
        <p>
          {recommendation.action === "review"
            ? "Automated outreach is disabled until real reply or meeting labels exist."
            : recommendation.best_window_start
              ? `${recommendation.best_window_start} → ${recommendation.best_window_end}`
              : "No contact window selected"}
        </p>
      </div>

      {recommendation.model_disagreement && (
        <div className="decision-warning">
          Models disagree materially. This account requires evidence review.
        </div>
      )}

      <div className="inspector-section">
        <div className="mini-label">ACCOUNT STORYCARD</div>
        <p className="story-copy">
          {recommendation.story_card?.summary ?? recommendation.reason}
        </p>
        <div className="evidence-note">
          <ShieldCheck size={14} />
          {recommendation.story_card?.generator ?? "deterministic"} · evidence
          locked
        </div>
      </div>

      <div className="inspector-section">
        <div className="mini-label">SCORE BREAKDOWN</div>
        <ScoreRow label="ICP fit" value={recommendation.icp_fit_score} />
        <ScoreRow
          label="Signal relevance"
          value={recommendation.signal_relevance_score}
        />
        <ScoreRow
          label="Future-event proxy"
          value={recommendation.proxy_event_score}
        />
        <ScoreRow
          label="Data confidence"
          value={recommendation.data_confidence_score}
        />
      </div>

      <div className="inspector-section">
        <div className="mini-label">RAW MODEL OUTPUTS</div>
        {recommendation.model_scores.map((score) => (
          <div className="model-row" key={score.model_id}>
            <span>{score.model_id}</span>
            <div className="score-track">
              <i style={{ width: `${score.score * 100}%` }} />
            </div>
            <b>{(score.score * 100).toFixed(0)}</b>
          </div>
        ))}
      </div>

      <div className="inspector-section">
        <div className="mini-label">SIGNAL TIMELINE</div>
        <div className="signal-timeline">
          {rawTrace === null ? (
            <div className="timeline-status">Loading account evidence…</div>
          ) : rawTrace.events.length === 0 ? (
            <div className="timeline-status">No recorded signal events.</div>
          ) : (
            rawTrace.events
              .slice()
              .sort(
                (left, right) =>
                  new Date(right.occurred_at).getTime() -
                  new Date(left.occurred_at).getTime(),
              )
              .slice(0, 6)
              .map((event) => (
                <div key={event.event_id}>
                  <i />
                  <span>
                    <b>{event.signal_type.replaceAll("_", " ")}</b>
                    <small>{event.kind.replaceAll("_", " ")}</small>
                  </span>
                  <time>{event.occurred_at.slice(0, 10)}</time>
                </div>
              ))
          )}
        </div>
      </div>

      <div className="inspector-section">
        <div className="mini-label">ACTION SIMULATION</div>
        {recommendation.action_evaluations.length === 0 ? (
          <div className="simulation-disabled">
            Disabled: response and timing models require real outreach outcomes.
          </div>
        ) : (
          recommendation.action_evaluations.map((evaluation) => (
            <div
              className={
                evaluation === best ? "simulation-row best" : "simulation-row"
              }
              key={`${evaluation.action}-${evaluation.wait_days}`}
            >
              <span>
                {evaluation.action === "wait"
                  ? `wait ${evaluation.wait_days}d`
                  : evaluation.action.replaceAll("_", " ")}
              </span>
              <b>{money(evaluation.expected_value)}</b>
            </div>
          ))
        )}
      </div>

      <div className="trace-counts">
        <div>
          <b>{rawTrace?.events?.length ?? "—"}</b>
          <span>EVENTS</span>
        </div>
        <div>
          <b>{rawTrace?.factsheets?.length ?? "—"}</b>
          <span>FACTSHEETS</span>
        </div>
      </div>
    </div>
  );
}
