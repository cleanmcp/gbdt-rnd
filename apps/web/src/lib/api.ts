import { z } from "zod";

const API_URL = "/api/engine";

export const runEventSchema = z.object({
  schema_version: z.literal(1),
  run_id: z.string(),
  seq: z.number().int().positive(),
  kind: z.string(),
  created_at: z.string(),
  payload: z.record(z.string(), z.unknown()),
});

export type RunEvent = z.infer<typeof runEventSchema>;

export type ProductContext = {
  product: {
    name: string;
    website: string;
    description: string;
    capabilities: string[];
    problems_solved: string[];
  };
  icp: {
    name: string;
    icp_version: string;
    industries: string[];
    operational_traits: string[];
    buyer_titles: string[];
  };
  goal: {
    name: string;
    objective: string;
    target_outcome: string;
  };
  signals: Array<{
    signal_type: string;
    description: string;
    shape: string;
    contract_hash: string;
  }>;
  accounts: number;
  events: number;
};

export type ModelScore = {
  model_id: string;
  score: number;
  uncertainty: number | null;
  top_factors: Array<[string, number]>;
};

export type Recommendation = {
  account_id: string;
  account_name: string | null;
  naics_code: string | null;
  state: string | null;
  rank: number;
  action: "contact_now" | "wait" | "never" | "review";
  best_window_start: string | null;
  best_window_end: string | null;
  opportunity_score: number;
  icp_fit_score: number;
  signal_relevance_score: number;
  proxy_event_score: number | null;
  data_confidence_score: number;
  model_disagreement: boolean;
  decision_status:
    "actionable" | "human_review" | "insufficient_evidence" | "research_only";
  reason: string;
  model_scores: ModelScore[];
  action_evaluations: Array<{
    action: string;
    wait_days: number;
    response_probability: number;
    expected_value: number;
  }>;
  story_card: {
    summary: string;
    implications: string[];
    confidence: number;
    generator: string;
  } | null;
};

export type AccountTrace = {
  account: {
    account_id: string;
    name: string;
    naics_code: string | null;
    state: string | null;
  };
  events: Array<{
    event_id: string;
    signal_type: string;
    kind: string;
    occurred_at: string;
  }>;
  factsheets: unknown[];
};

export type Benchmark = {
  model_id: string;
  label_kind: string;
  train_rows: number;
  test_rows: number;
  pr_auc: number | null;
  roc_auc: number | null;
  brier_score: number | null;
  precision_at_k: number | null;
  fit_seconds: number;
  score_seconds: number;
  status: "ok" | "unavailable" | "failed";
  detail: string | null;
};

export type Experiment = {
  run: {
    run_id: string;
    state: string;
    spec_hash: string;
    created_at: string;
    terminal_at: string | null;
    error_code: string | null;
    spec: {
      capacity: number;
      as_of: string;
      corpus_version: string;
      model_ids: string[];
      max_llm_story_cards: number;
    };
  };
  benchmarks: Benchmark[];
  recommendations: Recommendation[];
};

async function responseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(`${response.status}: ${await response.text()}`);
  }
  return (await response.json()) as T;
}

export async function readContext(): Promise<ProductContext> {
  return responseJson<ProductContext>(await fetch(`${API_URL}/v1/context`));
}

export async function createChatRun(
  message: string,
): Promise<{ run: Experiment["run"] }> {
  return responseJson(
    await fetch(`${API_URL}/v1/chat/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    }),
  );
}

export async function readExperiment(runId: string): Promise<Experiment> {
  return responseJson<Experiment>(
    await fetch(`${API_URL}/v1/experiments/${runId}`),
  );
}

export async function readAccount(accountId: string): Promise<AccountTrace> {
  return responseJson<AccountTrace>(
    await fetch(`${API_URL}/v1/accounts/${encodeURIComponent(accountId)}`),
  );
}

export function experimentStreamUrl(runId: string): string {
  return `${API_URL}/v1/experiments/${runId}/stream?after=0`;
}
