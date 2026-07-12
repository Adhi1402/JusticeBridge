// Mirrors the HTTP contract in justicebridge/api.py — keep in sync with
// AskRequest and _RESPONSE_FIELDS there.

export type Lang = "en" | "ta" | "hi" | "te";

export type Severity = "red" | "amber" | "green";

export interface AskRequest {
  text_input?: string;
  audio_base64?: string;
  images_base64?: string[];
  lang: Lang;
  want_tts?: boolean;
}

export interface Citation {
  act?: string;
  section_no?: string;
  title?: string;
}

export interface DlsaContact {
  name?: string;
  phone?: string;
  hours?: string;
  bring?: string;
  tele_law?: string;
}

export interface AgentTraceEntry {
  agent: string;
  duration_ms: number;
  status: "ok" | "error";
  output: Record<string, unknown>;
}

export interface AskResponse {
  transcript?: string;
  asr_confidence?: number;
  doc_text?: string;
  vision_confidence?: number;
  vertical?: string | null;
  kb_stores?: string[];
  supported?: boolean;
  planner_backend?: string;
  retrieval_sim?: number;
  citations?: Citation[];
  reasoning_backend?: string;
  grounded?: boolean;
  ungrounded_claims?: string[];
  severity?: Severity;
  deadline_days?: number | null;
  deadline_basis?: string | null;
  composite_confidence?: number;
  escalate?: boolean;
  eligibility_reasons?: string[];
  dlsa_contact?: DlsaContact | null;
  final_answer_en?: string;
  final_answer_local?: string;
  signal_packet?: Record<string, unknown>;
  lang?: Lang;
  error?: string[];
  audio_response_base64?: string;
  agent_trace?: AgentTraceEntry[];
}

export interface HealthResponse {
  llm_backend: string;
  llm_model?: string;
  llm_live: boolean;
  asr_backend: string;
  vision_backend: string;
  tts_backend: string;
  translation_backend?: string;
  offline?: boolean;
}

export interface KbStoreInfo {
  topic: string;
  description: string;
  cross_cutting: boolean;
}

export interface KbStoresResponse {
  supported: Record<string, KbStoreInfo>;
  coming_soon: Record<string, { topic: string }>;
}
