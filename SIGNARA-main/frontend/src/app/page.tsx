import Link from "next/link";

export default function Home() {
  return (
    <main className="mx-auto max-w-6xl px-6 py-16 text-slate-800">
      <section className="hero-shell glass-card-strong ambient-grid shine-border rounded-3xl p-8 sm:p-10">
        <div className="floating-orb orb-a" />
        <div className="floating-orb orb-b" />

        <p className="section-title stagger-in">Signara</p>
        <h1 className="stagger-in stagger-delay-1 mt-3 text-4xl font-bold sm:text-5xl">Bi-directional Sign Language Studio</h1>
        <p className="mt-4 max-w-3xl text-slate-600">
          Two dedicated experiences designed for speed, clarity, and real teaching workflows.
        </p>

        <div className="mt-6 flex flex-wrap gap-2 text-xs text-slate-600">
          <span className="premium-chip">Real-time webcam inference</span>
          <span className="premium-chip">Speech transcription with avatar playback</span>
          <span className="premium-chip">Confidence + latency debugging</span>
        </div>

        <div className="mt-10 grid gap-5 sm:grid-cols-2">
          <Link
            href="/speech-to-sign"
            className="glass-card lift-hover stagger-in rounded-2xl p-6"
          >
            <p className="text-xs uppercase tracking-widest text-sky-600">Mode 1</p>
            <h2 className="mt-2 text-2xl font-semibold">Speech → Sign Avatar</h2>
            <p className="mt-3 text-sm text-slate-600">Live transcript, speech recognition, and avatar playback for signs.</p>
            <p className="mt-4 text-sm font-semibold text-sky-700">Open mode →</p>
          </Link>

          <Link
            href="/sign-to-text"
            className="glass-card lift-hover stagger-in stagger-delay-1 rounded-2xl p-6"
          >
            <p className="text-xs uppercase tracking-widest text-cyan-700">Mode 2</p>
            <h2 className="mt-2 text-2xl font-semibold">Sign Detection</h2>
            <p className="mt-3 text-sm text-slate-600">Webcam-based ASL gloss detection with confidence, latency, and top predictions.</p>
            <p className="mt-4 text-sm font-semibold text-cyan-700">Open mode →</p>
          </Link>
        </div>
      </section>
    </main>
  );
}
