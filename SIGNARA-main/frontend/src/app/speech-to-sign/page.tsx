"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import AvatarViewer from "../../components/AvatarViewer";

type SpeechResult = { transcript: string };
type SpeechResultEntry = { isFinal: boolean; 0: SpeechResult };
type SpeechRecognitionEventLike = { resultIndex: number; results: SpeechResultEntry[] };

type SpeechRecognitionInstance = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};

type SpeechRecognitionConstructor = new () => SpeechRecognitionInstance;

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  }
}

export default function SpeechToSignPage() {
  const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [isSigning, setIsSigning] = useState(false);
  const [signs, setSigns] = useState<string[]>([]);
  const [currentAction, setCurrentAction] = useState("");
  const [lastApiSentText, setLastApiSentText] = useState("");
  const [speechMessage, setSpeechMessage] = useState("Idle");
  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null);
  const forceStopRef = useRef(false);

  const buildFallbackSigns = useCallback((text: string): string[] => {
    const knownGestures = new Set(["HELLO", "HI", "YES", "NO", "THANK", "THANKYOU", "YOU", "ILOVEYOU"]);
    return text
      .toUpperCase()
      .trim()
      .split(/\s+/)
      .flatMap((word) => {
        const cleanWord = word.replace(/[^A-Z]/g, "");
        if (!cleanWord) return [];
        if (knownGestures.has(cleanWord)) return [cleanWord];
        return cleanWord.split("");
      });
  }, []);

  const sendToBackend = useCallback(async (text: string) => {
    if (!text || text === lastApiSentText) return;
    setLastApiSentText(text);

    try {
      const response = await fetch(`${apiBaseUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error(`Predict request failed: ${response.status}`);
      }

      const data = await response.json();
      if (data.signs && data.signs.length > 0) {
        setSigns(data.signs);
      } else {
        setSigns(buildFallbackSigns(text));
      }
    } catch {
      setSigns(buildFallbackSigns(text));
    }
  }, [apiBaseUrl, buildFallbackSigns, lastApiSentText]);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setSpeechMessage("Speech recognition not supported in this browser");
      return;
    }

    if (!recognitionRef.current) {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = "en-US";

      recognition.onresult = (event: SpeechRecognitionEventLike) => {
        let finalTranscript = "";
        let interimTranscript = "";

        for (let i = event.resultIndex; i < event.results.length; i += 1) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          } else {
            interimTranscript += event.results[i][0].transcript;
          }
        }

        const fullTranscript = finalTranscript || interimTranscript;
        setTranscript(fullTranscript);
        if (finalTranscript.trim()) {
          setSpeechMessage("Recognized");
          void sendToBackend(finalTranscript.trim());
        }
      };

      recognition.onerror = (event: { error: string }) => {
        if (event.error === "network") {
          setSpeechMessage("Speech service network issue (use Chrome + internet)");
          setIsListening(false);
          return;
        }
        if (event.error === "not-allowed") {
          setSpeechMessage("Microphone access denied");
          setIsListening(false);
          return;
        }
        setSpeechMessage(`Speech error: ${event.error}`);
      };

      recognition.onend = () => {
        if (!forceStopRef.current && isListening) {
          try {
            recognition.start();
          } catch {
            setSpeechMessage("Speech restart failed");
          }
        }
      };

      recognitionRef.current = recognition;
    }

    if (isListening && recognitionRef.current) {
      forceStopRef.current = false;
      setSpeechMessage("Listening...");
      try {
        recognitionRef.current.start();
      } catch {
        setSpeechMessage("Speech engine busy, retrying");
      }
    }

    return () => {
      if (recognitionRef.current) {
        forceStopRef.current = true;
        recognitionRef.current.stop();
      }
    };
  }, [isListening, sendToBackend]);

  useEffect(() => {
    if (signs.length === 0) {
      setCurrentAction("");
      setIsSigning(false);
      return;
    }

    let i = 0;
    setIsSigning(true);
    setCurrentAction(signs[i]);

    const interval = setInterval(() => {
      i += 1;
      if (i >= signs.length) {
        setCurrentAction("");
        setIsSigning(false);
        clearInterval(interval);
        return;
      }
      setCurrentAction(signs[i]);
    }, 1200);

    return () => {
      clearInterval(interval);
      setIsSigning(false);
    };
  }, [signs]);

  return (
    <main className="mx-auto grid max-w-7xl gap-4 px-4 py-6 text-slate-800 lg:grid-cols-2">
      <section className="hero-shell glass-card-strong shine-border rounded-2xl p-6">
        <div className="floating-orb orb-a" />
        <div className="mb-4 flex items-center justify-between">
          <div>
            <p className="section-title">Mode</p>
            <h1 className="text-2xl font-bold">Speech → Sign Avatar</h1>
          </div>
          <span className="premium-chip">Voice Lab</span>
        </div>

          <div className="mb-3 grid grid-cols-2 gap-2 text-xs">
            <div className="metric-card p-2"><p className="text-slate-500">Recognizer</p><p className="font-semibold text-slate-700">Web Speech API</p></div>
            <div className="metric-card p-2"><p className="text-slate-500">Backend</p><p className="font-semibold text-slate-700">{apiBaseUrl}</p></div>
          </div>

          <button
            onClick={() => {
              if (isListening && recognitionRef.current) {
                forceStopRef.current = true;
                recognitionRef.current.stop();
                setIsListening(false);
                setSpeechMessage("Stopped");
              } else {
                setTranscript("");
                setIsListening(true);
              }
            }}
            className={`soft-pulse mx-auto flex h-28 w-28 items-center justify-center rounded-full text-5xl shadow-lg transition ${isListening ? "bg-rose-500" : "bg-cyan-500"}`}
          >
            🎤
          </button>
          <p className="mt-4 text-center text-sm text-slate-600">{speechMessage}</p>

          <div className="mt-4 min-h-40 rounded-xl border border-sky-100 bg-white/75 p-4 font-mono shadow-inner">
            {transcript || "Speech transcript appears here"}
          </div>
          <p className="mt-3 text-xs text-slate-500">Tip: if you see a network speech error, keep backend running and check internet/Chrome mic permissions.</p>
      </section>

      <section className="hero-shell glass-card-strong shine-border rounded-2xl p-4">
        <div className="floating-orb orb-b" />
        <div className="mb-3 flex items-center justify-between">
            <h2 className="text-lg font-semibold">Avatar Output</h2>
            <span className="premium-chip">{isSigning ? "Signing..." : "Idle"}</span>
        </div>
        <div className="h-[480px] overflow-hidden rounded-xl border border-sky-100 bg-slate-900">
            <AvatarViewer action={currentAction} />
        </div>
      </section>
    </main>
  );
}
