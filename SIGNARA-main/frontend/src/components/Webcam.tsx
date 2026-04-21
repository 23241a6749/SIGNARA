"use client";

import { useRef, useEffect, useState, useCallback } from "react";

interface WebcamComponentProps {
  onFrameCapture?: (frame: string) => void;
  width?: number;
  height?: number;
}

export default function WebcamComponent({ onFrameCapture, width = 640, height = 480 }: WebcamComponentProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width, height, facingMode: "user" },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          setIsStreaming(true);
        };
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start camera");
    }
  }, [width, height]);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  useEffect(() => {
    startCamera();
    return () => {
      stopCamera();
    };
  }, [startCamera, stopCamera]);

  return (
    <div className="relative rounded-lg overflow-hidden bg-black">
      <video
        ref={videoRef}
        width={width}
        height={height}
        className="w-full h-full object-cover"
        playsInline
        muted
      />
      <canvas ref={canvasRef} width={width} height={height} className="hidden" />
      
      {!isStreaming && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
          <div className="text-white text-center">
            <div className="animate-pulse mb-2">📹</div>
            <p>Click to enable camera</p>
          </div>
        </div>
      )}
      
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-red-900/50">
          <p className="text-white">{error}</p>
        </div>
      )}
    </div>
  );
}
