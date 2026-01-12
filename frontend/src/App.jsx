import { useMemo, useState } from "react";
import "./App.css";

const API_BASE = "/api";

function stripMarkdownFences(text) {
  // 简单去掉 ```lang 和 ```，让显示更干净（可选）
  return text.replace(/```[a-zA-Z]*\n?/g, "").replace(/```/g, "");
}

export default function App() {
  const [contextText, setContextText] = useState("");
  const [contextStatus, setContextStatus] = useState({ ok: false, chars: 0, msg: "" });

  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([
    { role: "assistant", content: "你好！先在左侧粘贴内容并保存，然后再提问，我会尽量基于你保存的内容回答。" },
  ]);
  const [retrievalInfo, setRetrievalInfo] = useState(null);
  const [useAllContexts, setUseAllContexts] = useState(true);
  const [loading, setLoading] = useState(false);

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading]);

  async function uploadContext() {
    const text = contextText.trim();
    if (!text) {
      setContextStatus({ ok: false, chars: 0, msg: "请先粘贴一些文本再保存。" });
      return;
    }

    setContextStatus({ ok: false, chars: 0, msg: "保存中..." });

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();

      if (!res.ok) {
        setContextStatus({ ok: false, chars: 0, msg: data?.detail ?? "保存失败" });
        return;
      }

      setContextStatus({ ok: true, chars: data.chars ?? text.length, msg: "已保存到后端内存 ✅" });
      setContextText("");
    } catch (e) {
      setContextStatus({ ok: false, chars: 0, msg: "请求失败：请确认后端在 8000 端口运行。" });
    }
  }

  async function send() {
    const text = input.trim();
    if (!text || loading) return;

    setMessages((m) => [...m, { role: "user", content: text }]);
    setInput("");
    setLoading(true);
    setRetrievalInfo(null);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, return_context: true, use_all_contexts: useAllContexts }),
      });

      const data = await res.json();
      const answer =
        data?.answer ??
        data?.choices?.[0]?.message?.content ??
        data?.choices?.[0]?.text ??
        "（无返回）";
      if (data?.retrieval) {
        setRetrievalInfo(data.retrieval);
      }
      setMessages((m) => [...m, { role: "assistant", content: stripMarkdownFences(answer) }]);
    } catch (e) {
      setMessages((m) => [...m, { role: "assistant", content: "请求失败：请检查后端是否启动，或 CORS 配置。" }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">AI Fullstack Starter</p>
          <h1>本地语义上下文 · 快速提问</h1>
          <p className="subtitle">FastAPI + Ollama 组合，让你的内容变成可对话的知识库。</p>
        </div>
        <div className="badges">
          <span className="badge">离线优先</span>
          <span className="badge">上下文检索</span>
          <span className="badge">轻量部署</span>
        </div>
      </header>

      <div className="grid">
        <section className="card card-left">
          <div className="card-header">
            <div>
              <p className="card-step">Step 01</p>
              <h2 className="card-title">粘贴你的内容</h2>
            </div>
            <span className={`status-dot ${contextStatus.ok ? "status-ok" : "status-idle"}`} />
          </div>

          <textarea
            className="context-input"
            value={contextText}
            onChange={(e) => setContextText(e.target.value)}
            placeholder="把你的笔记、文档片段或代码粘贴在这里..."
          />

          <div className="actions">
            <button className="primary-btn" onClick={uploadContext}>
              保存到后端
            </button>
            <div className={`status-text ${contextStatus.ok ? "status-ok-text" : ""}`}>
              {contextStatus.msg}
              {contextStatus.ok ? `（chars: ${contextStatus.chars}）` : ""}
            </div>
          </div>
        </section>

        <section className="card card-right">
          <div className="card-header">
            <div>
              <p className="card-step">Step 02</p>
              <h2 className="card-title">基于内容提问</h2>
            </div>
            <div className="chat-controls">
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={useAllContexts}
                  onChange={(e) => setUseAllContexts(e.target.checked)}
                />
                <span>检索全部历史</span>
              </label>
              <span className="hint-pill">{contextStatus.ok ? "Context 已就绪" : "等待保存"}</span>
            </div>
          </div>

          <div className="chat-window">
            {messages.length === 0 && !loading ? (
              <div className="empty-state">暂无对话记录，开始提问吧。</div>
            ) : (
              messages.map((msg, idx) => (
                <div key={idx} className={`message ${msg.role === "user" ? "user" : "assistant"}`}>
                  <div className="message-role">{msg.role === "user" ? "你" : "AI"}</div>
                  <div className="message-content">{msg.content}</div>
                </div>
              ))
            )}
            {loading && <div className="message assistant">AI：思考中...</div>}
          </div>

          <div className="composer">
            <input
              className="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && send()}
              placeholder="输入问题，比如：useState 是干嘛用的？"
            />
            <button className="ghost-btn" onClick={send} disabled={!canSend}>
              发送
            </button>
          </div>

          {retrievalInfo && (
            <div className="retrieval-panel">
              <div className="retrieval-title">
                命中段落（Top {retrievalInfo.top_k}）
              </div>
              <div className="retrieval-list">
                {retrievalInfo.chunks?.map((item, idx) => (
                  <div key={`${item.context_id ?? "ctx"}-${idx}`} className="retrieval-item">
                    <div className="retrieval-meta">
                      <span>匹配度：{item.score?.toFixed?.(3) ?? item.score}</span>
                      {item.context_id ? <span>批次：{item.context_id}</span> : null}
                    </div>
                    <div className="retrieval-text">{item.text}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="footer-tip">
            {contextStatus.ok ? "✅ 已保存 Context：回答将优先命中你的内容。" : "⚠️ 尚未保存 Context：回答将更泛化。"}
          </div>
        </section>
      </div>
    </div>
  );
}
