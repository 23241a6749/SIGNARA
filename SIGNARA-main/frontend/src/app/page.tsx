"use client";

import { useState, useEffect, useRef } from "react";
import WebcamComponent from "../components/Webcam";
import AvatarViewer from "../components/AvatarViewer";

// Add SpeechRecognition types
declare global {
  interface Window {
    SpeechRecognition: any;
    webkitSpeechRecognition: any;
  }
}

export default function Home() {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [isSigning, setIsSigning] = useState(false);
  const [signs, setSigns] = useState<string[]>([]);
  const [currentAction, setCurrentAction] = useState("");
  const [lastApiSentText, setLastApiSentText] = useState("");
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    if (!transcript) return;

    let isCancelled = false;

    const runSequence = async () => {
      const words = transcript
        .toUpperCase()
        .trim()
        .split(/\s+/)
        .map(word => word.replace(/[^A-Z]/g, ""));

      const knownGestures = ["HELLO", "HI", "YES", "NO", "THANK", "THANKYOU", "YOU", "ILOVEYOU"];

      for (const word of words) {
        if (!word) continue;

        if (knownGestures.includes(word)) {
          if (isCancelled) return;
          setCurrentAction(word);
          console.log("ACTION SQUENCE (WORD):", word);
          await new Promise(r => setTimeout(r, 1500));
        } else {
          // Spell the word out safely (fallback)
          for (const letter of word) {
            if (isCancelled) return;
            setCurrentAction(letter);
            console.log("ACTION SQUENCE (SPELL):", letter);
            await new Promise(r => setTimeout(r, 700));
          }
        }
      }

      if (!isCancelled) {
        setCurrentAction(""); // Clean reset after completion
      }
    };

    runSequence();

    return () => { isCancelled = true; };
  }, [transcript]);

  useEffect(() => {
    // Initialize Web Speech API
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      if (!recognitionRef.current) {
        recognitionRef.current = new SpeechRecognition();
        recognitionRef.current.continuous = true;
        recognitionRef.current.interimResults = true;
        recognitionRef.current.lang = 'en-US';

        recognitionRef.current.onresult = (event: any) => {
          let finalTranscript = "";
          let interimTranscript = "";

          for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
              finalTranscript += event.results[i][0].transcript;
            } else {
              interimTranscript += event.results[i][0].transcript;
            }
          }

          const fullTranscript = finalTranscript || interimTranscript;
          setTranscript(fullTranscript);

          if (finalTranscript.trim() !== '') {
            sendToBackend(finalTranscript.trim());
          }
        };

        recognitionRef.current.onerror = (event: any) => {
          console.error("Speech recognition error", event.error);
          if (event.error === 'not-allowed') {
            setIsListening(false);
          }
        };

        recognitionRef.current.onend = () => {
          // It only auto-restarts if it was still supposed to be listening (avoids bugs with stop tab)
          if (isListening && recognitionRef.current) {
            try {
              recognitionRef.current.start();
            } catch (e) { }
          }
        };
      }
    } else {
      console.warn("SpeechRecognition not supported in this browser.");
    }

    if (isListening && recognitionRef.current) {
      try {
        recognitionRef.current.start();
      } catch (e) { }
    } else if (!isListening && recognitionRef.current) {
      recognitionRef.current.stop();
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, [isListening]);

  // CREATE PLAY QUEUE SYSTEM
  useEffect(() => {
    if (signs.length === 0) return;

    let i = 0;

    const interval = setInterval(() => {
      setCurrentAction(signs[i]);
      i++;
      if (i > signs.length) {
        setCurrentAction(""); // Reset after finishing
        clearInterval(interval);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [signs]);

  const toggleListening = () => {
    setIsListening(!isListening);
    if (!isListening) {
      setTranscript("");
    }
  };

  const sendToBackend = async (text: string) => {
    if (!text || text === lastApiSentText) return;
    setLastApiSentText(text);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });

      const data = await response.json();

      if (data.signs && data.signs.length > 0) {
        setSigns(data.signs);
      }
    } catch (err) {
      console.error("Error calling predict:", err);
      // Fail safely
      setSigns([]);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      <header className="bg-black/30 backdrop-blur-md border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-4xl">🤟</span>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                Signara
              </h1>
              <p className="text-xs text-gray-400">Real-Time AI Speech-to-Sign Avatar</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 border border-white/10 bg-black/40 px-3 py-1.5 rounded-full">
              <div className={`w-3 h-3 rounded-full ${isListening ? "bg-red-500 animate-pulse" : "bg-gray-500"}`} />
              <span className="text-sm font-semibold text-gray-300">
                {isListening ? "Listening..." : isSigning ? "Signing..." : "Idle"}
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto p-4">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-[calc(100vh-100px)]">
          {/* Left Panel - Speech */}
          <div className="flex flex-col gap-4">
            <div className="bg-black/30 rounded-xl p-8 border border-white/10 flex flex-col items-center justify-center">
              <button
                onClick={toggleListening}
                className={`w-32 h-32 rounded-full flex items-center justify-center text-5xl transition-all shadow-xl ${isListening
                    ? "bg-red-500 hover:bg-red-600 shadow-red-500/50 scale-105"
                    : "bg-blue-500 hover:bg-blue-600 shadow-blue-500/50"
                  }`}
              >
                🎤
              </button>
              <h2 className="mt-6 text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-white to-gray-400">
                {isListening ? "Tap to Stop Listening" : "Tap to Speak"}
              </h2>
            </div>

            <div className="bg-black/30 rounded-xl p-6 border border-white/10 flex-1 flex flex-col shadow-inner">
              <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
                📜 Live Transcript
              </h2>
              <div className="flex-1 bg-black/50 rounded-lg p-5 font-mono text-xl overflow-y-auto border border-gray-700/50 relative">
                {transcript ? (
                  <p className="leading-relaxed whitespace-pre-wrap">{transcript}</p>
                ) : (
                  <div className="h-full flex items-center justify-center text-gray-500 italic text-center text-base">
                    Speech will appear here automatically when you talk.
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right Panel - Avatar */}
          <div className="lg:col-span-1">
            <div className="bg-black/30 rounded-xl p-4 border border-white/10 h-full relative overflow-hidden flex flex-col">
              <h2 className="text-lg font-semibold mb-3 z-10 flex items-center justify-between">
                <span>🧍 Teacher Avatar</span>
                {isSigning && <span className="text-xs bg-purple-500/30 text-purple-200 border border-purple-500/50 px-2 py-1 rounded-md animate-pulse">Action executing</span>}
              </h2>
              <div className="flex-1 rounded-xl overflow-hidden relative shadow-inner ring-1 ring-white/5">
                <AvatarViewer action={currentAction} />
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
