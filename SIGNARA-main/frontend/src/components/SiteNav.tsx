"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Home" },
  { href: "/speech-to-sign", label: "Speech → Sign" },
  { href: "/sign-to-text", label: "Sign Detection" },
];

export default function SiteNav() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-30 border-b border-sky-200/70 bg-white/75 backdrop-blur-xl">
      <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-4 py-3">
        <div className="stagger-in">
          <p className="text-[10px] uppercase tracking-[0.24em] text-sky-500">Signara Studio</p>
          <p className="text-sm font-semibold text-slate-700">Real-time ASL assistant</p>
        </div>

        <nav className="shine-border stagger-in stagger-delay-1 flex items-center gap-2 rounded-full border border-sky-100 bg-white/80 p-1 shadow-sm">
          {links.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`rounded-full px-3 py-1.5 text-sm transition ${
                  active
                    ? "bg-gradient-to-r from-sky-500 to-cyan-500 text-white shadow"
                    : "text-slate-500 hover:bg-sky-50 hover:text-sky-700"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
