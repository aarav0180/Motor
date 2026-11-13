/**
 * client/js/audio.js — Web Audio API Industrial Synthesizer Chime.
 * Plays high-priority audible warning chimes without external audio assets.
 */

class AudioManager {
  constructor() {
    this.audioCtx = null;
    this.isMuted = localStorage.getItem("vibra_muted") === "true";
  }

  _initContext() {
    if (!this.audioCtx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (AudioContextClass) {
        this.audioCtx = new AudioContextClass();
      }
    }
    if (this.audioCtx && this.audioCtx.state === "suspended") {
      this.audioCtx.resume();
    }
  }

  playAlertChime() {
    if (this.isMuted) return;

    try {
      this._initContext();
      if (!this.audioCtx) return;

      const now = this.audioCtx.currentTime;

      // Pulse 1: 880 Hz (A5)
      const osc1 = this.audioCtx.createOscillator();
      const gain1 = this.audioCtx.createGain();
      osc1.type = "sine";
      osc1.frequency.setValueAtTime(880, now);
      gain1.gain.setValueAtTime(0.15, now);
      gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.15);
      osc1.connect(gain1);
      gain1.connect(this.audioCtx.destination);
      osc1.start(now);
      osc1.stop(now + 0.15);

      // Pulse 2: 1174.66 Hz (D6) after 100ms
      const osc2 = this.audioCtx.createOscillator();
      const gain2 = this.audioCtx.createGain();
      osc2.type = "sine";
      osc2.frequency.setValueAtTime(1174.66, now + 0.1);
      gain2.gain.setValueAtTime(0.2, now + 0.1);
      gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
      osc2.connect(gain2);
      gain2.connect(this.audioCtx.destination);
      osc2.start(now + 0.1);
      osc2.stop(now + 0.35);
    } catch (e) {
      console.warn("[Audio] Could not play synthesized alert:", e);
    }
  }

  toggleMute() {
    this.isMuted = !this.isMuted;
    localStorage.setItem("vibra_muted", this.isMuted.toString());
    return this.isMuted;
  }
}

window.audioManager = new AudioManager();
