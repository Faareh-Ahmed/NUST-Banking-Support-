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
  
  // Upload & Modal states
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  const [error, setError] = useState("");
  const [isModalOpen, setIsModalOpen] = useState(false);

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

  async function handleUpload(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0] || null;
    if (!file || uploading) return;

    setError("");
    setUploadMessage("Uploading and indexing document... Please wait.");
    setUploading(true);

    try {
      const res = await uploadDocument(file);
      setUploadMessage(
        `Success! ${res.filename} indexed. Added chunks: ${res.indexed_chunks}.`
      );
      await refreshStats();
      
      // Auto-close modal after 3 seconds on success
      setTimeout(() => {
        setIsModalOpen(false);
        setUploadMessage("");
      }, 3000);
      
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Upload failed";
      setUploadMessage(`Error: ${msg}`);
      setError(msg);
    } finally {
      setUploading(false);
      // Allow uploading the same file again if desired
      e.target.value = "";
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
          <p>Make new policy or FAQ content searchable immediately.</p>
          <button 
            className="ghost-btn" 
            style={{ marginBottom: "1rem", outline: "1px solid #ccc" }} 
            type="button" 
            onClick={() => {
              setIsModalOpen(true);
              setUploadMessage("");
            }}>
            Upload Document
          </button>
          
          <button className="ghost-btn" type="button" onClick={refreshStats}>
            Refresh System Status
          </button>

          {isModalOpen && (
            <div style={{
              position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
              backgroundColor: "rgba(19, 32, 41, 0.75)", display: "flex",
              alignItems: "center", justifyContent: "center", zIndex: 9999,
              backdropFilter: "blur(4px)"
            }}>
              <div style={{
                maxWidth: "450px", width: "90%", backgroundColor: "var(--surface)", 
                padding: "2.5rem", borderRadius: "20px",
                boxShadow: "0 25px 50px rgba(0,0,0,0.25)", border: "1px solid var(--border)",
                display: "flex", flexDirection: "column", gap: "1rem", position: "relative"
              }}>
                <h2 style={{ margin: 0, color: "var(--brand-strong)", fontSize: "1.5rem" }}>Upload Knowledge Document</h2>
                <p style={{ margin: 0, color: "var(--muted)", lineHeight: 1.5 }}>
                  Add a new text or JSON file to the system. It will be indexed and available for chat immediately.
                </p>
                
                <div style={{
                  border: "2px dashed var(--border)", borderRadius: "12px", padding: "1.5rem", 
                  textAlign: "center", backgroundColor: "var(--bg-1)", margin: "0.5rem 0",
                  position: "relative", cursor: uploading ? "not-allowed" : "pointer",
                  transition: "all 0.2s"
                }}>
                  <span style={{ fontWeight: 600, color: "var(--brand)" }}>
                    {uploading ? "Processing file..." : "+ Click or drop file here"}
                  </span>
                  <input
                    type="file"
                    accept=".txt,.json"
                    onChange={handleUpload}
                    disabled={uploading}
                    style={{
                      position: "absolute", top: 0, left: 0, right: 0, bottom: 0, width: "100%", height: "100%", 
                      opacity: 0, cursor: uploading ? "not-allowed" : "pointer"
                    }}
                  />
                </div>
                
                {(uploading || uploadMessage) && (
                  <div style={{
                    padding: "0.75rem", borderRadius: "8px", fontSize: "0.9rem",
                    backgroundColor: error ? "#fef2f2" : "#ecfdf5",
                    color: error ? "var(--error)" : "var(--brand-strong)",
                    border: `1px solid ${error ? "#fecaca" : "#a7f3d0"}`
                  }}>
                    <strong style={{ display: "block", marginBottom: "4px" }}>
                      {error ? "Upload Error" : "System Status"}
                    </strong>
                    {uploadMessage}
                  </div>
                )}
                
                <button 
                  style={{ 
                    marginTop: "0.5rem", padding: "0.8rem 1.5rem", borderRadius: "8px",
                    backgroundColor: "transparent", color: "var(--muted)",
                    border: "1px solid var(--border)", cursor: uploading ? "not-allowed" : "pointer",
                    fontWeight: 600, fontSize: "1rem", transition: "0.2s"
                  }}
                  type="button" 
                  onClick={() => setIsModalOpen(false)}
                  disabled={uploading}
                >
                  {uploading ? "Please wait..." : "Cancel & Close"}
                </button>
              </div>
            </div>
          )}
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
                {/* Sources list deliberately removed as requested */}
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