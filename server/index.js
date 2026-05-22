/**
 * Assistant API (port 4000). Proxies speech/assistant to Python GPU server when USE_GPU_MODEL=1.
 */
import express from "express";
import { getAssistantResponse } from "../shared/assistant.js";

const port = Number(process.env.PORT) || 4000;
const useGPU = process.env.USE_GPU_MODEL === "1";
const gpuModelUrl = process.env.GPU_MODEL_URL || "http://127.0.0.1:5000";
const GPU_HEALTH_TIMEOUT_MS = Number(process.env.GPU_HEALTH_TIMEOUT_MS) || 90000;
const GPU_PROXY_TIMEOUT_MS = Number(process.env.GPU_PROXY_TIMEOUT_MS) || 180000;

const app = express();
/* Wav2Lip mp4 as base64 can be several MB */
app.use(express.json({ limit: "100mb" }));

class GpuHttpError extends Error {
  constructor(status, detail, body) {
    const msg =
      typeof detail === "string"
        ? detail
        : detail != null
          ? JSON.stringify(detail)
          : `GPU ${status}`;
    super(msg);
    this.name = "GpuHttpError";
    this.status = status;
    this.detail = detail;
    this.body = body;
  }
}

function parseGpuErrorBody(text) {
  if (!text) return null;
  try {
    const j = JSON.parse(text);
    if (j && j.detail != null) {
      return typeof j.detail === "object" ? j.detail : { reason: String(j.detail), detail: j.detail };
    }
    return j;
  } catch {
    return { reason: text, detail: text };
  }
}

async function fetchGpu(path, { method = "GET", body, timeoutMs } = {}) {
  const url = `${gpuModelUrl}${path}`;
  const ms = timeoutMs ?? (method === "GET" ? GPU_HEALTH_TIMEOUT_MS : GPU_PROXY_TIMEOUT_MS);
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), ms);
  try {
    const response = await fetch(url, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: ctrl.signal,
    });
    const text = await response.text();
    if (!response.ok) {
      const parsed = parseGpuErrorBody(text);
      const detail = parsed?.detail ?? parsed?.reason ?? parsed ?? text;
      throw new GpuHttpError(response.status, detail, parsed);
    }
    if (!text) return {};
    try {
      return JSON.parse(text);
    } catch {
      return { raw: text };
    }
  } catch (err) {
    if (err instanceof GpuHttpError) throw err;
    if (err && err.name === "AbortError") {
      throw new Error(`GPU request timed out after ${Math.round(ms / 1000)}s (${url})`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

async function proxyToGPU(path, body) {
  return fetchGpu(path, { method: "POST", body, timeoutMs: GPU_PROXY_TIMEOUT_MS });
}

function voiceErrorResponse(err, fallbackLabel) {
  if (err instanceof GpuHttpError) {
    const status = err.status >= 400 && err.status < 600 ? err.status : 502;
    const body =
      err.body && typeof err.body === "object"
        ? err.body
        : { ok: false, reason: String(err.detail || fallbackLabel), detail: err.detail };
    return { status, body };
  }
  return {
    status: 502,
    body: { ok: false, reason: fallbackLabel, detail: String(err?.message || err) },
  };
}

function gpuProxyMeta(extra = {}) {
  return {
    gpu_proxy: {
      enabled: useGPU,
      url: gpuModelUrl,
      health_timeout_ms: GPU_HEALTH_TIMEOUT_MS,
      proxy_timeout_ms: GPU_PROXY_TIMEOUT_MS,
      ...extra,
    },
  };
}

app.get("/", (_req, res) => {
  res.json({
    message: "Banking Operations Command Center API",
    gpu: useGPU,
    gpu_model_url: gpuModelUrl,
    routes: [
      "GET /api/health",
      "POST /api/speech",
      "POST /api/assistant",
      "POST /api/asr",
      "POST /api/voice/enroll",
      "POST /api/voice/identify",
      "POST /api/voice/passive",
    ],
  });
});

app.get("/api/health", async (_req, res) => {
  if (!useGPU) {
    return res.json({ status: "ok", gpu: false, mode: "local", ...gpuProxyMeta() });
  }
  try {
    const data = await fetchGpu("/health", { timeoutMs: GPU_HEALTH_TIMEOUT_MS });
    res.json({
      ...data,
      ...gpuProxyMeta({ connected: true }),
    });
  } catch (err) {
    const detail = String(err?.message || err);
    console.error("[noc] GPU health check failed:", detail);
    res.status(502).json({
      error: "GPU health check failed",
      detail,
      voice_id: { ready: false, backend: null, error: detail },
      ...gpuProxyMeta({ connected: false, error: detail }),
    });
  }
});

app.post("/api/assistant", async (req, res) => {
  try {
    const { command, parsed, text } = req.body || {};
    const payload = { text: text || command || "" };
    if (useGPU) {
      try {
        return res.json(await proxyToGPU("/assistant", payload));
      } catch (err) {
        console.warn("GPU assistant fallback:", err.message);
      }
    }
    res.json(getAssistantResponse({ command, parsed }));
  } catch (err) {
    res.status(500).json({ error: String(err) });
  }
});

app.post("/api/asr", async (req, res) => {
  if (!useGPU) {
    return res.status(503).json({ error: "GPU not enabled. Start with USE_GPU_MODEL=1" });
  }
  try {
    res.json(await proxyToGPU("/asr", req.body || {}));
  } catch (err) {
    const { status, body } = voiceErrorResponse(err, "asr_failed");
    console.error("asr proxy error:", err.message || err);
    res.status(status).json(body);
  }
});

app.post("/api/voice/enroll", async (req, res) => {
  if (!useGPU) {
    return res.status(503).json({ error: "GPU not enabled. Start with USE_GPU_MODEL=1" });
  }
  try {
    res.json(await proxyToGPU("/voice/enroll", req.body || {}));
  } catch (err) {
    const { status, body } = voiceErrorResponse(err, "voice_enroll_failed");
    console.error("voice enroll error:", err.message || err);
    res.status(status).json(body);
  }
});

app.post("/api/voice/identify", async (req, res) => {
  if (!useGPU) {
    return res.status(503).json({ error: "GPU not enabled. Start with USE_GPU_MODEL=1" });
  }
  try {
    res.json(await proxyToGPU("/voice/identify", req.body || {}));
  } catch (err) {
    const { status, body } = voiceErrorResponse(err, "voice_identify_failed");
    console.error("voice identify error:", err.message || err);
    res.status(status).json(body);
  }
});

app.post("/api/voice/passive", async (req, res) => {
  if (!useGPU) {
    return res.status(503).json({ error: "GPU not enabled. Start with USE_GPU_MODEL=1" });
  }
  try {
    res.json(await proxyToGPU("/voice/passive", req.body || {}));
  } catch (err) {
    const { status, body } = voiceErrorResponse(err, "voice_passive_failed");
    console.error("voice passive error:", err.message || err);
    res.status(status).json(body);
  }
});

app.post("/api/speech", async (req, res) => {
  const { text } = req.body || {};
  const line = String(text || "").trim();
  if (!line) {
    return res.status(400).json({ error: "text required" });
  }
  if (!useGPU) {
    return res.status(503).json({
      error: "GPU speech not enabled. Start with USE_GPU_MODEL=1",
    });
  }
  try {
    res.json(await proxyToGPU("/synthesize", { text: line }));
  } catch (err) {
    console.error("speech proxy error:", err.message);
    res.status(502).json({
      error: "GPU speech service unavailable",
      detail: err.message,
    });
  }
});

app.listen(port, () => {
  console.log(`Assistant API server listening on http://localhost:${port}`);
  console.log(`[noc] USE_GPU_MODEL=${useGPU ? "1" : "0"}`);
  console.log(`[noc] GPU_MODEL_URL=${gpuModelUrl}`);
  console.log(`[noc] GPU_HEALTH_TIMEOUT_MS=${GPU_HEALTH_TIMEOUT_MS}`);
  console.log(`[noc] GPU_PROXY_TIMEOUT_MS=${GPU_PROXY_TIMEOUT_MS}`);
  if (useGPU && (gpuModelUrl.includes("127.0.0.1") || gpuModelUrl.includes("localhost"))) {
    console.warn(
      "[noc] If modelServer.py runs on Kaggle, set GPU_MODEL_URL to your Kaggle tunnel URL (not 127.0.0.1)."
    );
  }
  if (useGPU) {
    console.log(`[noc] GPU proxy enabled → ${gpuModelUrl}`);
  }
});
