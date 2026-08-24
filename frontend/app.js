const API = "";
const POLL_INTERVAL_MS = 2500;
const POLL_TIMEOUT_MS = 90000;

const els = {
  chat: document.getElementById("chat"),
  emptyState: document.getElementById("emptyState"),
  messages: document.getElementById("messages"),
  composer: document.getElementById("composer"),
  input: document.getElementById("input"),
  sendBtn: document.getElementById("sendBtn"),
  chips: document.getElementById("chips"),
  themeToggle: document.getElementById("themeToggle"),
  ticker: document.getElementById("ticker"),
  tickerValue: document.getElementById("tickerValue"),
  tickerPct: document.getElementById("tickerPct"),
  messageTemplate: document.getElementById("messageTemplate"),
  revisionTemplate: document.getElementById("revisionTemplate"),
};

const session = { savedUsd: 0, baselineUsd: 0 };

// Full turn history sent with every request, since a context-free one-liner breaks any follow-up that references the prior turn.
const conversation = [];

// ---------- theme ----------

function applyTheme(theme) {
  if (theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  } else {
    document.documentElement.removeAttribute("data-theme");
    localStorage.removeItem("theme");
  }
}

const savedTheme = localStorage.getItem("theme");
if (savedTheme) applyTheme(savedTheme);

els.themeToggle.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const currentlyDark = current === "dark" || (!current && prefersDark);
  applyTheme(currentlyDark ? "light" : "dark");
});

// ---------- composer ----------

els.input.addEventListener("input", () => {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, 160) + "px";
});

els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    els.composer.requestSubmit();
  }
});

els.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = els.input.value.trim();
  if (!text) return;
  els.input.value = "";
  els.input.style.height = "auto";
  sendMessage(text);
});

els.chips.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  sendMessage(chip.dataset.prompt);
});

// ---------- message flow ----------

async function sendMessage(prompt) {
  if (els.emptyState) {
    els.emptyState.remove();
    els.emptyState = null;
  }

  appendUserMessage(prompt);
  conversation.push({ role: "user", content: prompt });

  const assistantEl = appendThinkingMessage();
  els.sendBtn.disabled = true;

  try {
    const res = await fetch(`${API}/v1/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: conversation }),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      renderError(assistantEl, body.detail || `Request failed (${res.status})`);
      return;
    }

    const data = await res.json();
    await renderAssistantMessage(assistantEl, data);
    updateTicker(data.routing.cost_usd, data.routing.baseline_cost_usd);

    // Patched in place if a revision arrives, so future turns reason from the corrected answer.
    const historyIndex = conversation.push({ role: "assistant", content: data.content }) - 1;
    pollForRevision(data.id, assistantEl, historyIndex);
  } catch (err) {
    renderError(assistantEl, "Couldn't reach the router. Is the API running?");
  } finally {
    els.sendBtn.disabled = false;
  }
}

function appendUserMessage(text) {
  const el = document.createElement("div");
  el.className = "message user";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  el.appendChild(bubble);
  els.messages.appendChild(el);
  scrollToBottom();
}

function appendThinkingMessage() {
  const node = els.messageTemplate.content.cloneNode(true);
  const el = node.querySelector(".message");
  el.classList.add("assistant");
  const bubble = el.querySelector(".bubble");
  bubble.innerHTML = '<span class="thinking"><span></span><span></span><span></span></span>';
  els.messages.appendChild(el);
  scrollToBottom();
  return el;
}

function renderError(el, message) {
  const bubble = el.querySelector(".bubble");
  bubble.classList.add("error");
  bubble.textContent = message;
}

async function renderAssistantMessage(el, data) {
  const bubble = el.querySelector(".bubble");
  bubble.textContent = "";
  await revealText(bubble, data.content);

  const chip = el.querySelector(".routing-chip");
  chip.hidden = false;
  chip.querySelector(".chip-model").textContent = data.routing.model_id;
  chip.querySelector(".chip-cost").textContent =
    data.routing.cost_usd === 0 ? "$0 (local)" : `$${data.routing.cost_usd.toFixed(5)}`;
  chip.querySelector(".chip-latency").textContent = `${Math.round(data.routing.latency_ms)}ms`;
  chip.querySelector(".details-reason").textContent = data.routing.reason;
  renderTierBars(chip.querySelector(".tier-bars"), data.routing.tier_probabilities, data.routing.tier);
  renderFeatures(chip.querySelector(".feature-grid"), data.routing.features);
  scrollToBottom();
}

function revealText(el, text) {
  // No real token streaming from the API, so this client-side reveal fakes it.
  return new Promise((resolve) => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches || text.length > 2000) {
      el.textContent = text;
      resolve();
      return;
    }
    const chunk = Math.max(1, Math.round(text.length / 90));
    let i = 0;
    const timer = setInterval(() => {
      i += chunk;
      el.textContent = text.slice(0, i);
      scrollToBottom();
      if (i >= text.length) {
        clearInterval(timer);
        resolve();
      }
    }, 12);
  });
}

const TIER_LABELS = { 1: "Tier 1", 2: "Tier 2", 3: "Tier 3" };

function renderTierBars(container, probabilities, activeTier) {
  container.innerHTML = "";
  for (const tier of [1, 2, 3]) {
    const p = probabilities[String(tier)] ?? 0;
    const row = document.createElement("div");
    row.className = "tier-row" + (tier === activeTier ? " active" : "");
    row.innerHTML = `
      <span class="tier-label">${TIER_LABELS[tier]}</span>
      <span class="tier-track"><span class="tier-fill" style="width:${Math.round(p * 100)}%"></span></span>
      <span class="tier-pct">${Math.round(p * 100)}%</span>
    `;
    container.appendChild(row);
  }
}

function renderFeatures(container, features) {
  container.innerHTML = "";
  const entries = Object.entries(features || {});
  for (const [key, value] of entries) {
    const row = document.createElement("div");
    row.className = "feature-row";
    const k = document.createElement("span");
    k.className = "key";
    k.textContent = key.replace(/_/g, " ");
    const v = document.createElement("span");
    v.className = "val";
    v.textContent = String(value);
    row.appendChild(k);
    row.appendChild(v);
    container.appendChild(row);
  }
}

function updateTicker(costUsd, baselineUsd) {
  session.savedUsd += baselineUsd - costUsd;
  session.baselineUsd += baselineUsd;
  els.ticker.hidden = false;
  els.tickerValue.textContent = `$${session.savedUsd.toFixed(4)}`;
  const pct = session.baselineUsd > 0 ? (session.savedUsd / session.baselineUsd) * 100 : 0;
  els.tickerPct.textContent = `(${pct.toFixed(0)}%)`;
}

// ---------- escalation polling ----------

function pollForRevision(requestId, messageEl, historyIndex) {
  const start = Date.now();
  const timer = setInterval(async () => {
    if (Date.now() - start > POLL_TIMEOUT_MS) {
      clearInterval(timer);
      return;
    }
    try {
      const res = await fetch(`${API}/v1/completions/${requestId}`);
      if (!res.ok) return;
      const status = await res.json();
      if (status.status === "pending") return;
      clearInterval(timer);
      if (status.status === "escalated" && status.escalated_content) {
        appendRevision(messageEl, status);
        // Future turns reason from the corrected answer, but the UI still shows both, honestly labeled.
        if (conversation[historyIndex]) {
          conversation[historyIndex].content = status.escalated_content;
        }
      }
    } catch {
      // network hiccup mid-poll — just stop, the original answer already stands
      clearInterval(timer);
    }
  }, POLL_INTERVAL_MS);
}

function appendRevision(messageEl, status) {
  const node = els.revisionTemplate.content.cloneNode(true);
  const el = node.querySelector(".revision");
  el.querySelector(".revision-bubble").textContent = status.escalated_content;
  const gap = status.quality_gap != null ? status.quality_gap.toFixed(1) : "?";
  const delta = status.cost_delta_usd != null ? `+$${status.cost_delta_usd.toFixed(5)}` : "";
  el.querySelector(".revision-meta").textContent = `quality gap ${gap}/5 on the original · ${delta} to verify`;
  messageEl.after(el);
  scrollToBottom();
}

function scrollToBottom() {
  els.chat.scrollTop = els.chat.scrollHeight;
}
