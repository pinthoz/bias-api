"use client";

import { useState, type ReactNode } from "react";

// Same origin by default: CloudFront forwards /predict to API Gateway and adds
// the API key there, so no key ever ships to the browser. For `npm run dev`,
// point this at the CloudFront URL (see .env.local.example).
const API_URL = process.env.NEXT_PUBLIC_API_URL || "/predict";

// Above this latency the request most likely hit a cold start
const COLD_START_MS = 2000;
const MAX_CHARS = 1000;

const EXAMPLES = [
  "Women are bad at math.",
  "The nurse finished her shift and went home.",
  "Old people can't learn new technology.",
];

type Category = "GEN" | "UNFAIR" | "STEREO";

const CATEGORIES: { id: Category; name: string; hint: string }[] = [
  { id: "GEN", name: "Generalisation", hint: "a blanket claim about a whole group" },
  { id: "UNFAIR", name: "Unfair language", hint: "disparaging or harsh wording about a group" },
  { id: "STEREO", name: "Stereotype", hint: "a trait attributed to a group" },
];

const COLOR: Record<Category, string> = {
  GEN: "var(--gen)",
  UNFAIR: "var(--unfair)",
  STEREO: "var(--stereo)",
};

// Response of POST /predict (see the API README)
type Token = {
  token: string;
  start: number;
  end: number;
  labels: string[];
  categories: Category[];
  scores: Record<string, number>;
};
type Span = { category: Category; start: number; end: number; score: number; text: string };
type Analysis = {
  biased: boolean;
  categories: Category[];
  biased_tokens: string[];
  spans: Span[];
  tokens: Token[];
};
type Prediction = { text: string; result: Analysis; ms: number };

export default function Home() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<Prediction[]>([]);

  async function analyse(input = text) {
    const value = input.trim();
    if (!value || loading) return;
    setLoading(true);
    setError(null);
    const t0 = performance.now();
    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: value }),
      });
      const ms = Math.round(performance.now() - t0);
      if (res.status === 429) throw new Error("Too many requests. Wait a second and try again.");
      if (res.status === 503)
        throw new Error("The model is starting up (cold start). Try again in a few seconds.");
      const data = await res.json().catch(() => null);
      if (!res.ok || !data) throw new Error(data?.error ?? `Request failed (${res.status})`);
      setHistory((h) => [{ text: value, result: data as Analysis, ms }, ...h].slice(0, 5));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  const latest = history[0];

  return (
    <main className="mx-auto w-full max-w-2xl px-4 py-12 sm:py-16">
      <header className="mb-8">
        <h1 className="text-3xl font-semibold tracking-tight">Bias Detector</h1>
        <p className="mt-2 text-muted">
          Type a sentence and a fine-tuned BERT model (GUS-Net) marks the words that carry social
          bias, and what kind of bias it is.
        </p>
      </header>

      <section className="rounded-xl border border-line bg-card p-5 shadow-sm">
        <label htmlFor="text" className="sr-only">
          Text to analyse
        </label>
        <textarea
          id="text"
          value={text}
          maxLength={MAX_CHARS}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyse();
          }}
          placeholder="Type or paste an English sentence…"
          className="min-h-32 w-full resize-y rounded-lg border border-line bg-bg p-3 outline-none focus:ring-2 focus:ring-accent"
        />

        <div className="mt-3 flex flex-wrap items-center gap-2">
          <button
            onClick={() => analyse()}
            disabled={loading || !text.trim()}
            className="rounded-lg bg-accent px-4 py-2 font-semibold text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Analysing…" : "Analyse"}
          </button>
          <button
            onClick={() => {
              setText("");
              setError(null);
            }}
            className="rounded-lg border border-line px-4 py-2 transition hover:bg-bg"
          >
            Clear
          </button>
          <span className="ml-auto text-sm text-muted">
            {text.length} / {MAX_CHARS} · Ctrl+Enter
          </span>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-2 text-sm text-muted">
          <span>Try:</span>
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              onClick={() => {
                setText(ex);
                analyse(ex);
              }}
              className="rounded-full border border-line px-3 py-1 transition hover:border-accent hover:text-ink"
            >
              {ex}
            </button>
          ))}
        </div>

        {error && (
          <p role="alert" className="mt-5 rounded-lg border border-warn/40 bg-warn/10 p-3 text-sm text-warn">
            {error}
          </p>
        )}

        {latest && !error && <Result prediction={latest} />}
      </section>

      {history.length > 1 && (
        <section className="mt-8">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted">Recent</h2>
          <ul className="divide-y divide-line rounded-xl border border-line bg-card">
            {history.slice(1).map((p, i) => (
              <li key={i} className="flex items-center gap-3 px-4 py-3 text-sm">
                <span className="flex shrink-0 gap-1" aria-hidden>
                  {p.result.categories.length ? (
                    p.result.categories.map((c) => (
                      <span key={c} className="h-2 w-2 rounded-full" style={{ background: COLOR[c] }} />
                    ))
                  ) : (
                    <span className="h-2 w-2 rounded-full bg-ok" />
                  )}
                </span>
                <span className="truncate">{p.text}</span>
                <span className="ml-auto shrink-0 font-mono text-xs text-muted">
                  {p.result.categories.join(" · ") || "no bias"} · {p.ms} ms
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}

      <footer className="mt-12 text-center text-xs text-muted">
        GUS-Net BERT → ONNX int8 → AWS Lambda (arm64) · API Gateway · S3 + CloudFront · Terraform
      </footer>
    </main>
  );
}

function Result({ prediction }: { prediction: Prediction }) {
  const { result, text, ms } = prediction;
  const cold = ms > COLD_START_MS;

  return (
    <div className="mt-6" aria-live="polite">
      <p className={`text-xl font-bold ${result.biased ? "text-warn" : "text-ok"}`}>
        {result.biased ? "Bias detected" : "No bias detected"}
      </p>

      <Highlighted text={text} tokens={result.tokens} />

      <ul className="mt-4 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
        {CATEGORIES.map((c) => (
          <li
            key={c.id}
            className={`flex items-center gap-1.5 ${result.categories.includes(c.id) ? "text-ink" : ""}`}
            title={c.hint}
          >
            <span className="h-1 w-4 rounded-full" style={{ background: COLOR[c.id] }} aria-hidden />
            {c.name}
          </li>
        ))}
      </ul>

      {result.spans.length > 0 && (
        <ul className="mt-5 space-y-2">
          {result.spans.map((s, i) => {
            const cat = CATEGORIES.find((c) => c.id === s.category);
            return (
              <li key={i} className="flex items-baseline gap-3 text-sm">
                <span
                  className="w-32 shrink-0 font-semibold"
                  style={{ color: COLOR[s.category] }}
                  title={cat?.hint}
                >
                  {cat?.name}
                </span>
                <span className="min-w-0 flex-1">“{s.text}”</span>
                <span className="shrink-0 font-mono text-xs text-muted">{(s.score * 100).toFixed(0)}%</span>
              </li>
            );
          })}
        </ul>
      )}

      <p className="mt-4 text-sm text-muted">
        {ms} ms{cold && " (likely a cold start)"}
      </p>
    </div>
  );
}

// Rebuilds the sentence from the API's character offsets, marking each flagged
// word with one underline stripe per category (hover shows the label scores)
function Highlighted({ text, tokens }: { text: string; tokens: Token[] }) {
  // Offsets count Unicode code points (Python), not UTF-16 units (JS)
  const chars = Array.from(text);
  const slice = (a: number, b: number) => chars.slice(a, b).join("");

  const parts: ReactNode[] = [];
  let pos = 0;
  tokens.forEach((t, i) => {
    if (t.start > pos) parts.push(<span key={`g${i}`}>{slice(pos, t.start)}</span>);
    const word = slice(t.start, t.end);
    if (t.categories.length) {
      parts.push(
        <mark
          key={i}
          className="rounded-sm px-0.5 text-ink"
          style={{
            background: `color-mix(in srgb, ${COLOR[t.categories[0]]} 14%, transparent)`,
            boxShadow: t.categories.map((c, k) => `0 ${2 + 3 * k}px 0 ${COLOR[c]}`).join(", "),
          }}
          title={Object.entries(t.scores)
            .map(([label, score]) => `${label} ${(score * 100).toFixed(0)}%`)
            .join(" · ")}
        >
          {word}
        </mark>,
      );
    } else {
      parts.push(<span key={i}>{word}</span>);
    }
    pos = t.end;
  });
  if (pos < chars.length) parts.push(<span key="tail">{slice(pos, chars.length)}</span>);

  return <p className="mt-4 text-lg leading-10">{parts}</p>;
}
