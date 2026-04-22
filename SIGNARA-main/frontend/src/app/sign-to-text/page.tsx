"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type TopPrediction = { gloss: string; confidence: number };

export default function SignToTextPage() {
  const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
  const wsBaseUrl = apiBaseUrl.replace(/^http/i, "ws");

  const [isStreaming, setIsStreaming] = useState(false);
  const [streamStatus, setStreamStatus] = useState("Disconnected");
  const [detectedGloss, setDetectedGloss] = useState("--");
  const [detectedConfidence, setDetectedConfidence] = useState(0);
  const [detectedLatency, setDetectedLatency] = useState(0);
  const [topPredictions, setTopPredictions] = useState<TopPrediction[]>([]);

  const webcamRef = useRef<HTMLVideoElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const captureTimerRef = useRef<number | null>(null);
  const sessionIdRef = useRef<string>("session");

  useEffect(() => {
    sessionIdRef.current = `session-${crypto.randomUUID()}`;
  }, []);

  const stopStreaming = useCallback(() => {
    if (captureTimerRef.current !== null) {
      window.clearInterval(captureTimerRef.current);
      captureTimerRef.current = null;
    }

    if (wsRef.current) {
      try {
        if (wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: "stop" }));
        }
      } catch {}
      wsRef.current.close();
      wsRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (webcamRef.current) {
      webcamRef.current.srcObject = null;
    }

    setIsStreaming(false);
    setStreamStatus("Disconnected");
  }, []);

  const startStreaming = useCallback(async () => {
    if (isStreaming) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
      });

      streamRef.current = stream;
      if (webcamRef.current) {
        webcamRef.current.srcObject = stream;
        await webcamRef.current.play();
      }

      setStreamStatus("Connecting...");
      const ws = new WebSocket(`${wsBaseUrl}/ws/stream/${sessionIdRef.current}`);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsStreaming(true);
        setStreamStatus("Connected");

        captureTimerRef.current = window.setInterval(() => {
          if (
            !webcamRef.current ||
            !captureCanvasRef.current ||
            ws.readyState !== WebSocket.OPEN
          ) {
            return;
          }

          const video = webcamRef.current;
          const canvas = captureCanvasRef.current;
          canvas.width = video.videoWidth || 640;
          canvas.height = video.videoHeight || 480;

          const ctx = canvas.getContext("2d");
          if (!ctx) return;
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          const image = canvas.toDataURL("image/jpeg", 0.7);
          ws.send(JSON.stringify({ type: "frame", image }));
        }, 180);
      };

      ws.onmessage = (event: MessageEvent<string>) => {
        try {
          const message = JSON.parse(event.data);
          if (message?.type !== "prediction") return;

          const data = message.data || {};
          const gloss = String(data.gloss || "--");
          const confidence = Number(data.confidence || 0);
          const latency = Number(data.latency_ms || 0);
          const buffering = Boolean(data.buffering);
          const top5 = Array.isArray(data.top5)
            ? data.top5.map((entry: [string, number]) => ({
                gloss: String(entry[0] || ""),
                confidence: Number(entry[1] || 0),
              }))
            : [];

          setDetectedGloss(gloss);
          setDetectedConfidence(confidence);
          setDetectedLatency(latency);
          setTopPredictions(top5);
          setStreamStatus(buffering ? "Buffering..." : "Connected");
        } catch {
          setStreamStatus("Connected");
        }
      };

      ws.onclose = () => {
        if (captureTimerRef.current !== null) {
          window.clearInterval(captureTimerRef.current);
          captureTimerRef.current = null;
        }
        setIsStreaming(false);
        setStreamStatus("Disconnected");
      };

      ws.onerror = () => {
        setStreamStatus("Socket error");
      };
    } catch {
      setStreamStatus("Camera permission denied");
      stopStreaming();
    }
  }, [isStreaming, stopStreaming, wsBaseUrl]);

  useEffect(() => {
    return () => {
      stopStreaming();
    };
  }, [stopStreaming]);

  return (
    <main className="mx-auto max-w-4xl px-4 py-6 text-slate-800">
      <section className="hero-shell glass-card-strong shine-border rounded-2xl p-4">
        <div className="floating-orb orb-a" />
        <div className="mb-3 flex items-center justify-between">
          <div>
            <p className="section-title">Mode</p>
            <h1 className="text-2xl font-bold">Sign Detection</h1>
          </div>
          <span className="premium-chip">Vision Lab</span>
        </div>

        <div className="mb-3 grid grid-cols-2 gap-2 text-xs sm:grid-cols-3">
          <div className="metric-card p-2"><p className="text-slate-500">Stream</p><p className="font-semibold text-slate-700">WebSocket</p></div>
          <div className="metric-card p-2"><p className="text-slate-500">Model</p><p className="font-semibold text-slate-700">WLASL v1</p></div>
          <div className="metric-card p-2"><p className="text-slate-500">Frame Rate</p><p className="font-semibold text-slate-700">~5 FPS</p></div>
        </div>

        <div className="mb-3 flex items-center justify-between">
            <h2 className="text-lg font-semibold">Camera Stream</h2>
            <button
              type="button"
              onClick={() => {
                if (isStreaming) {
                  stopStreaming();
                } else {
                  void startStreaming();
                }
              }}
            className={`premium-btn ${isStreaming ? "!bg-gradient-to-r !from-rose-500 !to-rose-400" : ""}`}
            >
              {isStreaming ? "Stop" : "Start"}
            </button>
        </div>

        <div className="aspect-video overflow-hidden rounded-xl border border-cyan-100 bg-slate-900">
            <video ref={webcamRef} autoPlay playsInline muted className="h-full w-full object-cover" />
            <canvas ref={captureCanvasRef} className="hidden" />
        </div>

        <div className="mt-3 grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
          <div className="metric-card p-2"><p className="text-slate-500">Status</p><p>{streamStatus}</p></div>
          <div className="metric-card p-2"><p className="text-slate-500">Gloss</p><p>{detectedGloss}</p></div>
          <div className="metric-card p-2"><p className="text-slate-500">Confidence</p><p>{(detectedConfidence * 100).toFixed(1)}%</p></div>
          <div className="metric-card p-2"><p className="text-slate-500">Latency</p><p>{detectedLatency.toFixed(0)} ms</p></div>
        </div>

        <div className="mt-3 rounded border border-cyan-100 bg-white/80 p-2 text-sm shadow-sm">
          <p className="mb-1 text-slate-500">Top predictions</p>
          {topPredictions.length === 0 ? (
            <p className="text-slate-500">No predictions yet</p>
          ) : (
            <p>{topPredictions.slice(0, 3).map((item) => `${item.gloss} ${(item.confidence * 100).toFixed(0)}%`).join(" | ")}</p>
          )}
        </div>
      </section>
    </main>
  );
}
