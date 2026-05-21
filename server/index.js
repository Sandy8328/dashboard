/**
 * Assistant API (port 4000). Proxies speech/assistant to Python GPU server when USE_GPU_MODEL=1.
 */
import express from "express";
import { getAssistantResponse } from "../shared/assistant.js";

const port = Number(process.env.PORT) || 4000;
const useGPU = process.env.USE_GPU_MODEL === "1";
const gpuModelUrl = process.env.GPU_MODEL_URL || "http://127.0.0.1:5000";

const app = express();
/* Wav2Lip mp4 as base64 can be several MB */
app.use(express.json({ limit: "100mb" }));

async function proxyToGPU(path, body) {
  const url = `${gpuModelUrl}${path}`;
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`GPU ${response.status}: ${detail}`);
  }
  return response.json();
}

app.get("/", (_req, res) => {
  res.json({
    message: "Banking Operations Command Center API",
    gpu: useGPU,
    routes: [
      "GET /api/health",
      "POST /api/speech",
      "POST /api/assistant",
      "POST /api/voice/enroll",
      "POST /api/voice/identify",
      "POST /api/voice/passive",
    ],
  });
});

app.get("/api/health", async (_req, res) => {
  if (!useGPU) {
    return res.json({ status: "ok", gpu: false, mode: "local" });
  }
  try {
    const r = await fetch(`${gpuModelUrl}/health`);
    const data = await r.json();
    res.json(data);
  } catch (err) {
    res.status(502).json({ error: "GPU health check failed", detail: String(err) });
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

app.post("/api/voice/enroll", async (req, res) => {
  if (!useGPU) {
    return res.status(503).json({ error: "GPU not enabled. Start with USE_GPU_MODEL=1" });
  }
  try {
    res.json(await proxyToGPU("/voice/enroll", req.body || {}));
  } catch (err) {
    console.error("voice enroll error:", err.message);
    res.status(502).json({ error: "Voice enroll failed", detail: err.message });
  }
});

app.post("/api/voice/identify", async (req, res) => {
  if (!useGPU) {
    return res.status(503).json({ error: "GPU not enabled. Start with USE_GPU_MODEL=1" });
  }
  try {
    res.json(await proxyToGPU("/voice/identify", req.body || {}));
  } catch (err) {
    console.error("voice identify error:", err.message);
    res.status(502).json({ error: "Voice identify failed", detail: err.message });
  }
});

app.post("/api/voice/passive", async (req, res) => {
  if (!useGPU) {
    return res.status(503).json({ error: "GPU not enabled. Start with USE_GPU_MODEL=1" });
  }
  try {
    res.json(await proxyToGPU("/voice/passive", req.body || {}));
  } catch (err) {
    console.error("voice passive error:", err.message);
    res.status(502).json({ error: "Voice passive failed", detail: err.message });
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
  if (useGPU) {
    console.log(`GPU proxy enabled; forwarding to ${gpuModelUrl}`);
  }
});
