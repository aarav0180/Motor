/**
 * client/sw.js — Service Worker for VibraGuard Web Client.
 * Handles background push notifications (RFC 8030 / VAPID) when the app is closed/dormant.
 */

self.addEventListener("install", (event) => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

// ---------------------------------------------------------------------------
// Background Push Notification Event
// ---------------------------------------------------------------------------
self.addEventListener("push", (event) => {
  let data = {
    title: "🚨 MOTOR VIBRATION ALERT",
    body: "Critical anomaly detected on Motor #1",
    data: { url: "/" },
  };

  if (event.data) {
    try {
      data = event.data.json();
    } catch (e) {
      data.body = event.data.text();
    }
  }

  const options = {
    body: data.body || "Anomaly threshold breached.",
    icon: data.icon || "/favicon.ico",
    badge: data.badge || "/favicon.ico",
    tag: data.tag || "motor-alert",
    renotify: true,
    requireInteraction: true,
    vibrate: [200, 100, 200, 100, 400],
    data: data.data || { url: "/" },
    actions: [
      { action: "review", title: "Review & Triage" },
      { action: "dismiss", title: "Dismiss" },
    ],
  };

  event.waitUntil(self.registration.showNotification(data.title, options));
});

// ---------------------------------------------------------------------------
// Notification Click Event (Focus or Open Dashboard)
// ---------------------------------------------------------------------------
self.addEventListener("notificationclick", (event) => {
  event.notification.close();

  if (event.action === "dismiss") {
    return;
  }

  const targetUrl = (event.notification.data && event.notification.data.url) || "/";

  event.waitUntil(
    self.clients.matchAll({ type: "window", includeUncontrolled: true }).then((clientList) => {
      // If a window is already open, focus it
      for (const client of clientList) {
        if ("focus" in client) {
          return client.focus();
        }
      }
      // Otherwise open a new window
      if (self.clients.openWindow) {
        return self.clients.openWindow(targetUrl);
      }
    })
  );
});
