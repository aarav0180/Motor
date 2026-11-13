/**
 * client/js/push.js — Web Push Notification Subscription & VAPID Key Manager.
 */

class PushManager {
  constructor() {
    this.swRegistration = null;
    this.isSupported = "serviceWorker" in navigator && "PushManager" in window;
    this.subscription = null;
  }

  async init() {
    if (!this.isSupported) {
      console.warn("[Push] Web Push is not supported in this browser.");
      return false;
    }

    try {
      this.swRegistration = await navigator.serviceWorker.register("/sw.js");
      this.subscription = await this.swRegistration.pushManager.getSubscription();
      return !!this.subscription;
    } catch (err) {
      console.error("[Push] Service Worker registration failed:", err);
      return false;
    }
  }

  urlBase64ToUint8Array(base64String) {
    const padding = "=".repeat((4 - (base64String.length % 4)) % 4);
    const base64 = (base64String + padding).replace(/-/g, "+").replace(/_/g, "/");
    const rawData = window.atob(base64);
    const outputArray = new Uint8Array(rawData.length);
    for (let i = 0; i < rawData.length; ++i) {
      outputArray[i] = rawData.charCodeAt(i);
    }
    return outputArray;
  }

  async subscribe() {
    if (!this.swRegistration) {
      await this.init();
    }
    if (!this.swRegistration) {
      throw new Error("Service Worker is not available.");
    }

    // 1. Fetch server's VAPID public key
    const { public_key } = await Api.getVapidPublicKey();
    const applicationServerKey = this.urlBase64ToUint8Array(public_key);

    // 2. Request browser permission and subscribe via pushManager
    const sub = await this.swRegistration.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey,
    });
    this.subscription = sub;

    // 3. Serialize and send to Tier 2 Server
    const subJson = sub.toJSON();
    const payload = {
      endpoint: subJson.endpoint,
      keys: {
        p256dh: subJson.keys.p256dh,
        auth: subJson.keys.auth,
      },
      client_id: `browser_${navigator.platform}_${Date.now()}`,
    };

    await Api.subscribePush(payload);
    return true;
  }

  async unsubscribe() {
    if (!this.subscription) return true;

    const endpoint = this.subscription.endpoint;
    await this.subscription.unsubscribe();
    this.subscription = null;

    try {
      await Api.unsubscribePush(endpoint);
    } catch (e) {
      console.warn("[Push] Unsubscribe endpoint notification error:", e);
    }
    return true;
  }

  hasSubscription() {
    return !!this.subscription;
  }
}

window.pushManager = new PushManager();
