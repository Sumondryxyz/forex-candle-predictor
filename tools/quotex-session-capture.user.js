// ==UserScript==
// @name         Quotex session capture
// @namespace    forex-candle-predictor
// @version      1.0.0
// @description  Log in to Quotex once; this saves the session to the predictor bot on your phone.
// @match        https://market-qx.trade/*
// @match        https://qxbroker.com/*
// @match        https://*.market-qx.trade/*
// @grant        GM_cookie
// @grant        GM_xmlhttpRequest
// @connect      127.0.0.1
// @connect      localhost
// @run-at       document-idle
// ==/UserScript==

/*
 * How it works
 * ------------
 * Cloudflare blocks the Quotex login page from datacenter IPs, so the phone
 * logs in and this script forwards the resulting session to a tiny local
 * server (`python scripts/session_capture.py`), which writes `session.json`.
 *
 * The token comes from the authenticated `/api/v1/cabinets/digest` endpoint.
 * The cookies must include HttpOnly ones (`laravel_session`, `cf_clearance`),
 * which `document.cookie` cannot see — so we read them with Tampermonkey's
 * `GM_cookie` API and fall back to `document.cookie` if it is unavailable.
 *
 * No password is ever read. Only the session the browser already holds.
 */

(function () {
  'use strict';

  const CAPTURE_URL = 'http://127.0.0.1:8787/save';
  const CAPTURE_KEY = ''; // set the same value as CAPTURE_KEY on the server, if you use one
  const POLL_MS = 4000;
  const REFRESH_MS = 30 * 60 * 1000;

  let savedToken = null;

  function badge(text, ok) {
    let el = document.getElementById('qx-capture-badge');
    if (!el) {
      el = document.createElement('div');
      el.id = 'qx-capture-badge';
      el.style.cssText =
        'position:fixed;z-index:2147483647;right:10px;bottom:10px;padding:8px 12px;' +
        'border-radius:9px;font:600 13px system-ui,sans-serif;color:#fff;' +
        'box-shadow:0 4px 14px rgba(0,0,0,.35);transition:opacity .3s';
      document.body.appendChild(el);
    }
    el.style.background = ok ? '#16a34a' : '#b45309';
    el.textContent = text;
    el.style.opacity = '1';
    if (ok) setTimeout(() => { el.style.opacity = '0'; }, 4000);
  }

  async function getToken() {
    try {
      const r = await fetch('/api/v1/cabinets/digest', { credentials: 'include' });
      if (!r.ok) return null;
      const j = await r.json();
      const d = j && j.data ? j.data : j;
      return d.token || d.ssid || null;
    } catch (e) {
      return null;
    }
  }

  // GM_cookie sees HttpOnly cookies; document.cookie does not.
  function gmCookies() {
    return new Promise((resolve) => {
      if (typeof GM_cookie === 'undefined' || !GM_cookie.list) return resolve([]);
      try {
        GM_cookie.list({ domain: location.hostname }, (cookies, err) => {
          resolve(!err && Array.isArray(cookies) ? cookies : []);
        });
      } catch (e) {
        resolve([]);
      }
    });
  }

  function cookieString(list) {
    if (list && list.length) {
      return list.map((c) => c.name + '=' + c.value).join('; ');
    }
    return document.cookie || '';
  }

  function post(body) {
    return new Promise((resolve, reject) => {
      if (typeof GM_xmlhttpRequest === 'function') {
        GM_xmlhttpRequest({
          method: 'POST',
          url: CAPTURE_URL,
          headers: { 'Content-Type': 'application/json' },
          data: JSON.stringify(body),
          onload: (r) => resolve(r),
          onerror: (e) => reject(e),
        });
      } else {
        fetch(CAPTURE_URL, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        }).then(resolve, reject);
      }
    });
  }

  async function capture() {
    const token = await getToken();
    if (!token) return false;
    if (token === savedToken) return true;

    const cookies = cookieString(await gmCookies());
    if (!cookies) {
      badge('Quotex: logged in but no cookies read', false);
      return false;
    }

    const body = {
      token: token,
      cookies: cookies,
      user_agent: navigator.userAgent,
      key: CAPTURE_KEY,
    };
    try {
      const res = await post(body);
      if (res.status >= 200 && res.status < 300) {
        savedToken = token;
        badge('Session saved \u2713', true);
        return true;
      }
      badge('Quotex capture rejected (' + res.status + ')', false);
    } catch (e) {
      badge('Quotex capture failed \u2014 is the server running?', false);
    }
    return false;
  }

  // Quotex is a single-page app: login happens without a reload, so poll.
  let ticks = 0;
  const timer = setInterval(async () => {
    ticks += 1;
    const done = await capture();
    if (done || ticks > 150) clearInterval(timer); // stop after ~10 min or success
  }, POLL_MS);

  // Re-save periodically so a long-lived tab keeps the session fresh.
  setInterval(() => { capture(); }, REFRESH_MS);

  capture();
})();
