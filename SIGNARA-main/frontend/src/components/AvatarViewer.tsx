"use client";

import React from "react";

const gestureMap: Record<string, string> = {
  HELLO: "/gestures/hello.png",
  HI: "/gestures/hi.png",
  YES: "/gestures/yes.png",
  NO: "/gestures/no.png",
  YOU: "/gestures/you.png",
  THANK: "/gestures/thankyou.png",
  THANKYOU: "/gestures/thankyou.png",
};

interface GestureDisplayProps {
  action?: string;
}

export default function AvatarViewer({ action = "" }: GestureDisplayProps) {
  const imgAction = action.toUpperCase();
  
  let img = gestureMap[imgAction];
  if (!img && imgAction.length === 1 && /^[A-Z]$/.test(imgAction)) {
    // All A-Z letters now have generated PNG images
    const allLetters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    if (allLetters.includes(imgAction)) {
      img = `/gestures/${imgAction}.png`;
    }
  }

  return (
    <div className="flex justify-center items-center w-full h-full bg-gradient-to-b from-gray-800 to-gray-900 rounded-lg relative overflow-hidden ring-4 ring-black/20">
      {action && img ? (
        <img
          key={imgAction} // Forces re-render animation if same image triggers
          src={img}
          alt={imgAction}
          className="w-72 h-72 object-contain animate-pulse drop-shadow-[0_0_25px_rgba(255,255,255,0.2)]"
        />
      ) : action ? (
        <div className="flex flex-col items-center justify-center opacity-40">
           <p className="text-gray-400 font-semibold text-2xl tracking-widest uppercase">No gesture for "{action}"</p>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center opacity-40">
          <span className="text-7xl mb-4 grayscale">💬</span>
          <p className="text-gray-400 font-semibold text-lg tracking-widest uppercase">Awaiting Speech</p>
        </div>
      )}
      
      {action && (
        <div className="absolute bottom-6 left-0 right-0 flex justify-center pointer-events-none">
          <div className="bg-black/80 backdrop-blur-md text-white px-8 py-4 rounded-full border border-white/10 shadow-[0_10px_40px_rgba(0,0,0,0.5)]">
            <span className="text-4xl font-extrabold tracking-widest text-[#A8E6CF] uppercase drop-shadow-md">
              {action}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
