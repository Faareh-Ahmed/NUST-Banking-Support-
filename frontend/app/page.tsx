"use client";

import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";
import { getStats, sendChat, uploadDocument } from "@/lib/api";

type UiMessage = {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  latencyMs?: number;
  guardrailTriggered?: boolean;
  outOfDomain?: boolean;
};

const WELCOME_MESSAGE = `Welcome to NUST Bank AI Support.\n\nI can help you with account types, banking products, cards, app features, and policy questions.`;

export default function Home() {
  const [messages, setMessages] = useState<UiMessage[]>([
    { role: "assistant", content: WELCOME_MESSAGE },
  ]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [statsText, setStatsText] = useState("Loading system status...");
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    refreshStats();
  }, []);

  async function refreshStats() {
    try {
      const stats = await getStats();
      setStatsText(
        `${stats.indexed_documents} indexed chunks | LLM: ${stats.llm_model} | Embeddings: ${stats.embedding_model}`,
      );
    } catch {
      setStatsText("Backend is not reachable. Start FastAPI server on port 8000.");
    }
  }

  async function handleSend(e: FormEvent) {
    e.preventDefault();
    const message = input.trim();
    if (!message || isSending) return;

    setError("");
    setInput("");
    setMessages((prev: UiMessage[]) => [...prev, { role: "user", content: message }]);
    setIsSending(true);

    try {
      const response = await sendChat(message);
      setMessages((prev: UiMessage[]) => [
        ...prev,
        {
          role: "assistant",
          content: response.answer,
          sources: response.sources,
          latencyMs: response.latency_ms,
          guardrailTriggered: response.guardrail_triggered,
          outOfDomain: response.out_of_domain,
        },
      ]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to send message";
      setError(msg);
    } finally {
      setIsSending(false);
    }
  }

  async function handleUpload(file: File | null) {
    if (!file || uploading) return;

    setError("");
    setUploadMessage("");
    setUploading(true);

    try {
      const res = await uploadDocument(file);
      setUploadMessage(
        `${res.filename} indexed successfully. Added/updated chunks: ${res.indexed_chunks}. Total indexed chunks: ${res.indexed_documents_total}.`,
      );
      await refreshStats();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Upload failed";
      setError(msg);
    } finally {
      setUploading(false);
    }
  }

  const chatTitle = useMemo(() => (isSending ? "Thinking..." : "Ready"), [isSending]);

  return (
    <main className="page">
      <section className="hero">
        <p className="kicker">NUST Bank Digital Assistant</p>
        <h1>Customer Support That Stays Grounded In Your Bank Knowledge Base</h1>
        <p className="status">{statsText}</p>
      </section>

      <section className="layout-grid">
        <aside className="card">
          <h2>Knowledge Base Updates</h2>
          <p>Upload a .txt or .json document to make new policy or FAQ content searchable immediately.</p>
          <label className="upload-control">
            <span>Select Document</span>
            <input
              type="file"
              accept=".txt,.json"
              onChange={(e: ChangeEvent<HTMLInputElement>) => handleUpload(e.target.files?.[0] || null)}
              disabled={uploading}
            />
          </label>
          <p className="small">{uploading ? "Indexing document..." : uploadMessage || "No recent uploads"}</p>
          <button className="ghost-btn" type="button" onClick={refreshStats}>
            Refresh System Status
          </button>
        </aside>

        <section className="card chat-card">
          <div className="chat-header">
            <h2>Live Support Chat</h2>
            <span className="pill">{chatTitle}</span>
          </div>

          <div className="chat-window">
            {messages.map((m, idx) => (
              <article key={`${m.role}-${idx}`} className={`msg ${m.role}`}>
                <p>{m.content}</p>
                {m.sources && m.sources.length > 0 && (
                  <p className="meta">Sources: {m.sources.join(", ")}</p>
                )}
                {typeof m.latencyMs === "number" && <p className="meta">Latency: {m.latencyMs}ms</p>}
                {m.guardrailTriggered && <p className="warn">Safety filter applied</p>}
                {m.outOfDomain && <p className="meta">Out-of-domain query detected</p>}
              </article>
            ))}
          </div>

          <form className="chat-form" onSubmit={handleSend}>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about NUST Bank products, accounts, cards, policies, or app features..."
              rows={3}
              disabled={isSending}
            />
            <button type="submit" disabled={isSending || !input.trim()}>
              {isSending ? "Sending..." : "Send Message"}
            </button>
          </form>

          {error && <p className="error">{error}</p>}
        </section>
      </section>
    </main>
  );
}
