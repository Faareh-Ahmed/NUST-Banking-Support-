export type ChatResponse = {
  answer: string;
  sources: string[];
  latency_ms: number;
  guardrail_triggered: boolean;
  out_of_domain: boolean;
};

export type StatsResponse = {
  indexed_documents: number;
  llm_model: string;
  embedding_model: string;
  total_queries: number;
  avg_latency_ms: number;
  guardrail_triggers: number;
  out_of_domain_count: number;
};

export type UploadResponse = {
  filename: string;
  indexed_chunks: number;
  indexed_documents_total: number;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8000";

export async function getStats(): Promise<StatsResponse> {
  const res = await fetch(`${API_BASE_URL}/stats`, { cache: "no-store" });
  if (!res.ok) throw new Error("Unable to fetch stats");
  return res.json();
}

export async function sendChat(message: string): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Chat request failed");
  }
  return res.json();
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const fd = new FormData();
  fd.append("file", file);

  const res = await fetch(`${API_BASE_URL}/upload`, {
    method: "POST",
    body: fd,
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Upload failed");
  }
  return res.json();
}
