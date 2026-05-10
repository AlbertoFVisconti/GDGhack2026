import { useCallback, useRef, useState, useEffect } from "react";

export function useSpeech() {
  const [isMuted, setIsMuted] = useState(false);
  const queueRef = useRef<string[]>([]);
  const isSpeakingRef = useRef(false);

  const processQueue = useCallback(() => {
    if (isSpeakingRef.current || queueRef.current.length === 0 || isMuted) {
      return;
    }
    const text = queueRef.current.shift();
    if (!text) return;

    isSpeakingRef.current = true;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US"; // Forces American English pronunciation
    utterance.rate = 0.8; // 1.0 is default, 0.9 slows it down slightly
    utterance.pitch = 1.0; // 1.0 is default, adjust if you want it deeper/higher
    utterance.onend = () => {
      isSpeakingRef.current = false;
      processQueue();
    };
    utterance.onerror = () => {
      isSpeakingRef.current = false;
      processQueue();
    };
    window.speechSynthesis.speak(utterance);
  }, [isMuted]);

  const speak = useCallback(
    (text: string, urgent: boolean = false) => {
      if (isMuted) return;

      if (urgent) {
        window.speechSynthesis.cancel();
        queueRef.current = [];
        isSpeakingRef.current = false;
      }

      queueRef.current.push(text);
      processQueue();
    },
    [isMuted, processQueue],
  );

  const toggleMute = useCallback(() => {
    setIsMuted((prev) => {
      const next = !prev;
      if (next) {
        window.speechSynthesis.cancel();
        queueRef.current = [];
        isSpeakingRef.current = false;
      }
      return next;
    });
  }, []);

  useEffect(() => {
    return () => {
      window.speechSynthesis.cancel();
    };
  }, []);

  return { speak, isMuted, toggleMute };
}
