const express = require('express');
const https = require('https');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;
const POLYGON_KEY = process.env.POLYGON_KEY || 'cBE5Kbq9yllt0Yj29mDQjBcIKfAYQlHF';
const POLY = 'https://api.polygon.io';
const STATE_FILE = path.join(__dirname, 'state.json');

// ═══════════════════════════════════════════════════════
// STRATEGY CONFIG
// ═══════════════════════════════════════════════════════
const CFG = {
  pt: 1.65,
  gapDn: 1.5,
  gapUp: 0.75,
  threshold: 0.75,
  comm: 1.10,
  tickers: {
    QQQ: { delta: 0.55, budget: 104000 },
    SPY: { delta: 0.60, budget: 99000 },
  },
};

// ═══════════════════════════════════════════════════════
// SHARED STATE — single source of truth for all clients
// ═══════════════════════════════════════════════════════
let STATE = {
  date: null, // YYYY-MM-DD this state is for
  lastUpdate: null,
  tickers: {
    QQQ: emptyTicker('QQQ'),
    SPY: emptyTicker('SPY'),
  },
};

function emptyTicker(tk) {
  return {
    signal: false,
    loc: 0,
    priorC: null, priorH: null, priorL: null,
    open: null, live: null, gapUp: null, budget: null,
    strike: null, locked: false, optTicker: null,
    bid: null, ask: null, mid: null, cts: null, delta: null,
    ptHit: false, ptHitTime: null,
    entryPx: null, entryTime: null,
    exitPx: null, exitTime: null, exitReason: null,
    phase: 'pre',
  };
}

// ═══════════════════════════════════════════════════════
// PERSISTENCE — survives Railway redeploys
// ═══════════════════════════════════════════════════════
function saveState() {
  try {
    fs.writeFileSync(STATE_FILE, JSON.stringify(STATE, null, 2));
  } catch (e) {
    console.error('Failed to save state:', e.message);
  }
}

function loadState() {
  try {
    if (fs.existsSync(STATE_FILE)) {
      const saved = JSON.parse(fs.readFileSync(STATE_FILE, 'utf8'));
      if (saved.date === getToday()) {
        STATE = saved;
        console.log(`Restored state for ${saved.date}`);
        return;
      }
    }
  } catch (e) {
    console.error('Failed to load state:', e.message);
  }
  // Fresh day — reset
  STATE = { date: getToday(), lastUpdate: null, tickers: { QQQ: emptyTicker('QQQ'), SPY: emptyTicker('SPY') } };
}

// ═══════════════════════════════════════════════════════
// TIME HELPERS — all in Eastern
// ═══════════════════════════════════════════════════════
function getET() {
  return new Date(new Date().toLocaleString('en-US', { timeZone: 'America/New_York' }));
}

function getToday() {
  const et = getET();
  return et.getFullYear() + '-' + String(et.getMonth() + 1).padStart(2, '0') + '-' + String(et.getDate()).padStart(2, '0');
}

function getETMinutes() {
  const et = getET();
  return et.getHours() * 60 + et.getMinutes();
}

function getPhase() {
  const et = getET();
  const dow = et.getDay();
  if (dow === 0 || dow === 6) return 'closed';
  const m = et.getHours() * 60 + et.getMinutes();
  if (m < 9 * 60 + 30) return 'pre';
  if (m < 9 * 60 + 32) return 'entry';
  if (m < 9 * 60 + 50) return 'active';
  if (m < 9 * 60 + 51) return 'time_exit';
  if (m < 16 * 60) return 'post';
  return 'closed';
}

function etTimeString() {
  const et = getET();
  return et.toLocaleTimeString('en-US', { hour12: false });
}

// ═══════════════════════════════════════════════════════
// POLYGON API
// ═══════════════════════════════════════════════════════
function polyFetch(path) {
  return new Promise((resolve, reject) => {
    const sep = path.includes('?') ? '&' : '?';
    const url = POLY + path + sep + 'apiKey=' + POLYGON_KEY;
    https.get(url, { timeout: 10000 }, (res) => {
      let data = '';
      res.on('data', (chunk) => (data += chunk));
      res.on('end', () => {
        try { resolve(JSON.parse(data)); }
        catch (e) { resolve({}); }
      });
    }).on('error', () => resolve({}));
  });
}

// ═══════════════════════════════════════════════════════
// DATA FETCHERS
// ═══════════════════════════════════════════════════════
async function fetchSignals() {
  const today = getToday();
  if (STATE.date !== today) {
    STATE = { date: today, lastUpdate: null, tickers: { QQQ: emptyTicker('QQQ'), SPY: emptyTicker('SPY') } };
  }

  const start = new Date(Date.now() - 10 * 86400000).toISOString().slice(0, 10);
  for (const tk of ['QQQ', 'SPY']) {
    const d = await polyFetch(`/v2/aggs/ticker/${tk}/range/1/day/${start}/${today}?adjusted=true&sort=asc&limit=15`);
    const bars = d.results || [];
    const prior = bars.filter((b) => new Date(b.t).toISOString().slice(0, 10) < today).slice(-1)[0];
    if (!prior) continue;
    const rng = prior.h - prior.l;
    STATE.tickers[tk].loc = rng > 0 ? (prior.c - prior.l) / rng : 0.5;
    STATE.tickers[tk].priorC = prior.c;
    STATE.tickers[tk].priorH = prior.h;
    STATE.tickers[tk].priorL = prior.l;
    STATE.tickers[tk].signal = STATE.tickers[tk].loc > CFG.threshold;
  }
  saveState();
}

async function fetchPrices() {
  for (const tk of ['QQQ', 'SPY']) {
    const d = await polyFetch(`/v2/snapshot/locale/us/markets/stocks/tickers/${tk}`);
    const snap = d.ticker;
    if (!snap) continue;
    STATE.tickers[tk].live = snap.lastTrade?.p || snap.day?.c;
    const open = snap.day?.o;
    if (open) STATE.tickers[tk].open = open;
    const eff = open || STATE.tickers[tk].live;
    if (eff && STATE.tickers[tk].priorC) {
      STATE.tickers[tk].gapUp = eff > STATE.tickers[tk].priorC;
      const mult = STATE.tickers[tk].gapUp ? CFG.gapUp : CFG.gapDn;
      STATE.tickers[tk].budget = Math.round(CFG.tickers[tk].budget * mult / 1000) * 1000;
    }
  }
}

async function fetchOptions() {
  const today = getToday();
  const p = getPhase();
  const lock = ['entry', 'active', 'time_exit', 'post'].includes(p);

  for (const tk of ['QQQ', 'SPY']) {
    const stk = STATE.tickers[tk];
    if (!stk.signal) continue;
    const tgt = CFG.tickers[tk].delta;
    const ul = stk.live || stk.open;
    if (!ul) continue;

    const is0DTE = (o) => o.details?.expiration_date === today;
    const priceOK = (b, a) => (b == null || a == null) ? true : (b + a) / 2 < 30;

    if (stk.locked && stk.strike) {
      // Already locked — just refresh quotes
      const d = await polyFetch(`/v3/snapshot/options/${tk}?expiration_date=${today}&option_type=put&strike_price_gte=${stk.strike}&strike_price_lte=${stk.strike}&limit=10`);
      const best = (d.results || []).filter(is0DTE)[0];
      if (best) {
        const b = best.last_quote?.bid ?? null;
        const a = best.last_quote?.ask ?? null;
        if (priceOK(b, a)) {
          stk.bid = b;
          stk.ask = a;
          const newMid = (b != null && a != null) ? (b + a) / 2 : (best.day?.close ?? stk.mid);
          stk.mid = newMid;
        }
        stk.delta = best.greeks?.delta ? Math.abs(best.greeks.delta) : stk.delta;
        if (!stk.optTicker) stk.optTicker = best.details?.ticker || null;
      }
    } else {
      // Discovery phase — find best strike
      const d = await polyFetch(`/v3/snapshot/options/${tk}?expiration_date=${today}&option_type=put&strike_price_gte=${Math.floor(ul * 0.95)}&strike_price_lte=${Math.ceil(ul * 1.12)}&limit=250&order=desc`);
      const results = (d.results || []).filter((o) => {
        if (!o.greeks?.delta || !is0DTE(o)) return false;
        const b = o.last_quote?.bid, a = o.last_quote?.ask;
        return !(b != null && a != null && (b + a) / 2 > 30);
      });
      let best = null, bestD = 99;
      for (const o of results) {
        const dd = Math.abs(Math.abs(o.greeks.delta) - tgt);
        if (dd < bestD) { bestD = dd; best = o; }
      }
      if (best) {
        stk.strike = best.details?.strike_price;
        stk.delta = Math.abs(best.greeks.delta);
        stk.bid = best.last_quote?.bid ?? null;
        stk.ask = best.last_quote?.ask ?? null;
        stk.mid = (stk.bid != null && stk.ask != null) ? (stk.bid + stk.ask) / 2 : (best.day?.close ?? null);
        stk.optTicker = best.details?.ticker || null;
      }

      // Lock at entry
      if (lock && stk.strike && !stk.locked) {
        stk.locked = true;
        stk.entryPx = stk.mid;
        stk.entryTime = etTimeString();
        stk.cts = stk.mid && stk.budget ? Math.floor(stk.budget / (stk.mid * 100)) : null;
        console.log(`LOCKED ${tk}: strike=${stk.strike} entry=${stk.entryPx} cts=${stk.cts} @ ${stk.entryTime}`);
      }
    }

    // Update contracts if not set
    if (stk.mid && stk.budget && !stk.cts) {
      stk.cts = Math.floor(stk.budget / (stk.mid * 100));
    }

    // Check PT hit
    if (stk.locked && stk.entryPx && stk.mid && !stk.ptHit) {
      if (stk.mid >= stk.entryPx * CFG.pt) {
        stk.ptHit = true;
        stk.ptHitTime = etTimeString();
        stk.exitPx = stk.entryPx * CFG.pt;
        stk.exitTime = stk.ptHitTime;
        stk.exitReason = 'profit_target';
        console.log(`PT HIT ${tk}: exit=${stk.exitPx.toFixed(4)} @ ${stk.ptHitTime}`);
      }
    }

    // Check time exit at 9:50
    if (stk.locked && stk.entryPx && !stk.ptHit && !stk.exitReason && p === 'time_exit') {
      stk.exitPx = stk.mid;
      stk.exitTime = etTimeString();
      stk.exitReason = 'time_exit_950';
      console.log(`TIME EXIT ${tk}: exit=${stk.exitPx?.toFixed(4)} @ ${stk.exitTime}`);
    }
  }
  saveState();
}

// ═══════════════════════════════════════════════════════
// SSE — push state to all connected clients
// ═══════════════════════════════════════════════════════
const clients = new Set();

function broadcast() {
  const payload = JSON.stringify({
    state: STATE,
    phase: getPhase(),
    time: etTimeString(),
    today: getToday(),
    cfg: CFG,
  });
  const msg = `data: ${payload}\n\n`;
  for (const res of clients) {
    try { res.write(msg); }
    catch (e) { clients.delete(res); }
  }
}

// ═══════════════════════════════════════════════════════
// POLLING LOOP
// ═══════════════════════════════════════════════════════
let signalsFetched = false;

async function poll() {
  try {
    const today = getToday();
    if (STATE.date !== today) {
      STATE = { date: today, lastUpdate: null, tickers: { QQQ: emptyTicker('QQQ'), SPY: emptyTicker('SPY') } };
      signalsFetched = false;
      saveState();
    }

    if (!signalsFetched) {
      await fetchSignals();
      signalsFetched = true;
    }

    const phase = getPhase();
    if (phase !== 'closed') {
      await fetchPrices();
      if (['pre', 'entry', 'active', 'time_exit'].includes(phase)) {
        await fetchOptions();
      }
    }

    STATE.lastUpdate = new Date().toISOString();
    broadcast();
  } catch (e) {
    console.error('Poll error:', e.message);
  }
}

// ═══════════════════════════════════════════════════════
// ROUTES
// ═══════════════════════════════════════════════════════
app.get('/events', (req, res) => {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
    'Access-Control-Allow-Origin': '*',
  });

  // Send current state immediately
  const payload = JSON.stringify({
    state: STATE,
    phase: getPhase(),
    time: etTimeString(),
    today: getToday(),
    cfg: CFG,
  });
  res.write(`data: ${payload}\n\n`);

  clients.add(res);
  req.on('close', () => clients.delete(res));
});

app.get('/state', (req, res) => {
  res.json({
    state: STATE,
    phase: getPhase(),
    time: etTimeString(),
    today: getToday(),
    cfg: CFG,
  });
});

app.get('/health', (req, res) => {
  res.json({ ok: true, date: getToday(), phase: getPhase(), clients: clients.size });
});

// Serve frontend
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});
app.use(express.static(path.join(__dirname, 'public')));

// ═══════════════════════════════════════════════════════
// START
// ═══════════════════════════════════════════════════════
loadState();

// Start listening FIRST so Railway health check passes, THEN start polling
app.listen(PORT, () => {
  console.log(`Dashboard server running on port ${PORT}`);
  console.log(`State date: ${STATE.date}, phase: ${getPhase()}`);
});

// Delay initial poll to let the server fully start
setTimeout(() => {
  console.log('Starting initial poll...');
  poll().then(() => console.log('Initial poll complete'));

  // Poll every 15 seconds
  setInterval(() => poll(), 15000);

  // Re-fetch signals every 5 minutes (in case of late data)
  setInterval(() => { signalsFetched = false; }, 5 * 60 * 1000);
}, 2000);
