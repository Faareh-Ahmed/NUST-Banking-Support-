"use client";

import { ChangeEvent, FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import { getStats, sendChat, uploadDocument } from "@/lib/api";

type Message = {
  role: "user" | "assistant";
  content: string;
  latencyMs?: number;
};

const WELCOME: Message = {
  role: "assistant",
  content:
    "Hello! I'm the NUST Bank virtual assistant.\n\nI can help you with account information, banking products, cards, mobile app features, loans, and insurance. What can I help you with today?",
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([WELCOME]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [docCount, setDocCount] = useState<number | null>(null);
  const [online, setOnline] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");
  const [uploadOk, setUploadOk] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => { loadStats(); }, []);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending]);

  async function loadStats() {
    try {
      const s = await getStats();
      setDocCount(s.indexed_documents);
      setOnline(true);
    } catch {
      setOnline(false);
    }
  }

  async function submit(e: FormEvent) {
    e.preventDefault();
    const msg = input.trim();
    if (!msg || sending) return;

    setInput("");
    setMessages(p => [...p, { role: "user", content: msg }]);
    setSending(true);

    try {
      const res = await sendChat(msg);
      setMessages(p => [...p, {
        role: "assistant",
        content: res.answer,
        latencyMs: res.latency_ms,
      }]);
    } catch {
      setMessages(p => [...p, {
        role: "assistant",
        content: "I'm having trouble connecting right now. Please try again shortly.",
      }]);
    } finally {
      setSending(false);
    }
  }

  function handleKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit(e as unknown as FormEvent);
    }
  }

  async function handleUpload(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file || uploading) return;

    setUploadMsg("Uploading and processing document...");
    setUploadOk(true);
    setUploading(true);

    try {
      const res = await uploadDocument(file);
      setUploadMsg(`"${res.filename}" added successfully — ${res.indexed_chunks} passages indexed.`);
      setDocCount(res.indexed_documents_total);
      setTimeout(() => { setModalOpen(false); setUploadMsg(""); }, 2800);
    } catch {
      setUploadMsg("Upload failed. Please check the file format and try again.");
      setUploadOk(false);
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  }

  function openModal() { setModalOpen(true); setUploadMsg(""); setUploadOk(true); }
  function closeModal() { if (!uploading) setModalOpen(false); }

  return (
    <div className="shell">

      {/* ── Top bar ───────────────────────────────────────────── */}
      <header className="topbar">
        <div className="topbar-left">
          <div className="logo">
            <span className="logo-mark">N</span>
          </div>
          <div className="topbar-name">
            <span className="bank-name">NUST Bank</span>
            <span className="bank-sub">Customer Support</span>
          </div>
        </div>
        <div className="topbar-right">
          <span className={`dot ${online ? "dot--on" : "dot--off"}`} />
          <span className="online-label">{online ? "Assistant online" : "Connecting…"}</span>
        </div>
      </header>

      {/* ── Main ──────────────────────────────────────────────── */}
      <main className="body">

        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-block">
            <p className="sidebar-eyebrow">I can help with</p>
            <ul className="topic-list">
              {[
                "Savings & current accounts",
                "Loans and finance products",
                "Debit & credit cards",
                "Mobile app & fund transfers",
                "Bancassurance & insurance",
              ].map(t => (
                <li key={t}>
                  <svg className="check-icon" viewBox="0 0 16 16" fill="none">
                    <circle cx="8" cy="8" r="7.25" stroke="currentColor" strokeWidth="1.5" />
                    <polyline points="5,8.5 7,10.5 11,6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  {t}
                </li>
              ))}
            </ul>
          </div>

          <div className="sidebar-sep" />

          <div className="sidebar-block">
            <p className="sidebar-eyebrow">Knowledge base</p>
            <p className="kb-count">
              {docCount !== null
                ? <><strong>{docCount.toLocaleString()}</strong> documents available</>
                : "Loading…"}
            </p>
            <button className="btn-upload" onClick={openModal}>
              <UploadIcon />
              Add document
            </button>
          </div>


        </aside>

        {/* Chat */}
        <section className="chat">
          <div className="chat-feed">
            {messages.map((m, i) => (
              <div key={i} className={`row row--${m.role}`}>
                {m.role === "assistant" && <div className="av">N</div>}
                <div className={`bubble bubble--${m.role}`}>
                  <p>{m.content}</p>
                  {m.latencyMs !== undefined && (
                    <span className="timing">{(m.latencyMs / 1000).toFixed(1)}s</span>
                  )}
                </div>
              </div>
            ))}

            {sending && (
              <div className="row row--assistant">
                <div className="av">N</div>
                <div className="bubble bubble--assistant bubble--typing">
                  <span /><span /><span />
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <form className="chat-form" onSubmit={submit}>
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKey}
              placeholder="Ask about accounts, cards, loans, or app features…"
              rows={2}
              disabled={sending}
            />
            <button type="submit" disabled={sending || !input.trim()} className="btn-send" aria-label="Send">
              <SendIcon />
            </button>
          </form>
        </section>
      </main>

      {/* ── Upload modal ──────────────────────────────────────── */}
      {modalOpen && (
        <div className="overlay" onClick={e => e.target === e.currentTarget && closeModal()}>
          <div className="modal" role="dialog" aria-modal="true">
            <button className="modal-x" onClick={closeModal} aria-label="Close">
              <CloseIcon />
            </button>

            <div className="modal-icon-wrap"><UploadIcon size={26} /></div>
            <h2 className="modal-title">Add to Knowledge Base</h2>
            <p className="modal-desc">
              Upload a document to expand the assistant's knowledge. Changes take effect immediately.
            </p>

            <label className={`drop-zone${uploading ? " drop-zone--busy" : ""}`}>
              <input type="file" accept=".txt,.json" onChange={handleUpload} disabled={uploading} />
              <span className="dz-label">
                {uploading ? "Processing…" : "Click to browse or drop a file"}
              </span>
              <span className="dz-hint">Supported formats: .txt &nbsp;·&nbsp; .json</span>
            </label>

            {uploadMsg && (
              <p className={`upload-status ${uploadOk ? "upload-status--ok" : "upload-status--err"}`}>
                {uploadMsg}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Inline SVG icons ──────────────────────────────────────────────────── */

function UploadIcon({ size = 15 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}
