/* ============================================================
   OAQ-SR — project page interactions
   ============================================================ */
(function () {
  'use strict';

  var C = {
    dense:   '#ff7a4d',
    outlier: '#4cc2ff',
    muted:   '#7b8699',
    line:    '#262d3d',
    text:    '#e9ecf3',
    good:    '#6ee7a8'
  };

  /* -------------------------------------------------- scroll reveal ---- */
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
    });
  }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
  document.querySelectorAll('.reveal').forEach(function (el) { io.observe(el); });

  /* -------------------------------------------------- before/after ----- */
  document.querySelectorAll('.ba').forEach(function (ba) {
    var after = ba.querySelector('.after');
    var handle = ba.querySelector('.handle');
    var dragging = false;

    function set(clientX) {
      var r = ba.getBoundingClientRect();
      var p = Math.max(0, Math.min(1, (clientX - r.left) / r.width));
      after.style.clipPath = 'inset(0 0 0 ' + (p * 100) + '%)';
      handle.style.left = (p * 100) + '%';
    }
    ba.addEventListener('pointerdown', function (e) {
      dragging = true; ba.setPointerCapture(e.pointerId); set(e.clientX);
    });
    ba.addEventListener('pointermove', function (e) {
      if (dragging || e.pointerType === 'mouse') set(e.clientX);
    });
    ba.addEventListener('pointerup', function () { dragging = false; });
    ba.addEventListener('pointercancel', function () { dragging = false; });
  });

  /* ================================================================
     Piecewise-linear-quantizer playground
     ================================================================ */
  var canvas = document.getElementById('plq-canvas');
  if (canvas) initPlayground(canvas);

  function initPlayground(canvas) {
    var ctx = canvas.getContext('2d');

    /* --- a deterministic, skewed activation distribution ------------- */
    var seed = 20250101;
    function rnd() {                       // mulberry32
      seed |= 0; seed = seed + 0x6D2B79F5 | 0;
      var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
    function gauss() {
      var u = 1 - rnd(), v = rnd();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
    var SAMPLES = [];
    for (var i = 0; i < 24000; i++) {
      var r = rnd(), x;
      if (r < 0.955) x = gauss() * 26 - 4;                    // dense bulk
      else if (r < 0.982) x = -90 - Math.abs(gauss()) * 62;   // heavy negative tail
      else x = 70 + Math.abs(gauss()) * 52;                   // lighter positive tail
      SAMPLES.push(x);
    }
    var LO = Infinity, HI = -Infinity;
    for (var k = 0; k < SAMPLES.length; k++) {
      if (SAMPLES[k] < LO) LO = SAMPLES[k];
      if (SAMPLES[k] > HI) HI = SAMPLES[k];
    }

    /* --- histogram --------------------------------------------------- */
    var NB = 150, hist = new Float64Array(NB), binW = (HI - LO) / NB;
    SAMPLES.forEach(function (v) {
      var b = Math.min(NB - 1, Math.floor((v - LO) / binW));
      hist[b]++;
    });
    var histMax = 0;
    for (var h0 = 0; h0 < NB; h0++) if (hist[h0] > histMax) histMax = hist[h0];

    /* --- quantizers -------------------------------------------------- */
    function quantUniform(v, lo, hi, levels) {
      var x = Math.max(lo, Math.min(hi, v));
      var s = (hi - lo) / (levels - 1);
      return Math.round((x - lo) / s) * s + lo;
    }
    function pointsUniform(lo, hi, levels) {
      var s = (hi - lo) / (levels - 1), out = [];
      for (var i = 0; i < levels; i++) out.push(lo + i * s);
      return out;
    }
    var SORTED = SAMPLES.slice().sort(function (a, b) { return a - b; });
    function percentile(p) {
      return SORTED[Math.min(SORTED.length - 1, Math.floor(p * SORTED.length))];
    }
    var P_LO = percentile(0.005), P_HI = percentile(0.995);

    function evaluate(mode, bits, bpPct) {
      var levels = Math.pow(2, bits), qp = [], quant;

      if (mode === 'minmax') {
        qp = pointsUniform(LO, HI, levels);
        quant = function (v) { return quantUniform(v, LO, HI, levels); };

      } else if (mode === 'percentile') {
        qp = pointsUniform(P_LO, P_HI, levels);
        quant = function (v) { return quantUniform(v, P_LO, P_HI, levels); };

      } else {                                    /* ours: piecewise linear */
        var bp = percentile(bpPct);
        var half = Math.pow(2, bits - 1);
        qp = qp.concat(pointsUniform(-bp, bp, levels))
               .concat(pointsUniform(LO, -bp, half))
               .concat(pointsUniform(bp, HI, half));
        quant = function (v) {
          if (v < -bp) return quantUniform(v, LO, -bp, half);
          if (v > bp)  return quantUniform(v, bp, HI, half);
          return quantUniform(v, -bp, bp, levels);
        };
      }

      var mse = 0;
      for (var i = 0; i < SAMPLES.length; i++) {
        var d = SAMPLES[i] - quant(SAMPLES[i]);
        mse += d * d;
      }
      return { mse: mse / SAMPLES.length, qp: qp };
    }

    /* --- drawing ----------------------------------------------------- */
    var state = { mode: 'ours', bits: 4, bpPct: 0.99 };

    function draw() {
      var dpr = Math.min(2, window.devicePixelRatio || 1);
      var W = canvas.clientWidth, H = Math.round(W * 0.46);
      canvas.width = W * dpr; canvas.height = H * dpr;
      canvas.style.height = H + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      var padL = 10, padR = 10, padT = 14, padB = 40;
      var plotW = W - padL - padR, plotH = H - padT - padB;
      var X = function (v) { return padL + (v - LO) / (HI - LO) * plotW; };

      var res = evaluate(state.mode, state.bits, state.bpPct);
      var bp = state.mode === 'ours' ? percentile(state.bpPct) : null;

      /* histogram bars, coloured by region */
      for (var b = 0; b < NB; b++) {
        var c0 = LO + b * binW, c1 = c0 + binW, mid = (c0 + c1) / 2;
        var h = Math.pow(hist[b] / histMax, 0.42) * plotH;
        var inDense = bp === null ? true : (mid >= -bp && mid <= bp);
        ctx.fillStyle = inDense ? 'rgba(255,122,77,.85)' : 'rgba(76,194,255,.62)';
        ctx.fillRect(X(c0), padT + plotH - h, Math.max(1, X(c1) - X(c0) - 0.6), h);
      }

      /* baseline */
      var yBase = padT + plotH;
      ctx.strokeStyle = C.line; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(padL, yBase + .5); ctx.lineTo(W - padR, yBase + .5); ctx.stroke();

      /* breakpoints */
      if (bp !== null) {
        [-bp, bp].forEach(function (v) {
          ctx.strokeStyle = 'rgba(233,236,243,.55)'; ctx.lineWidth = 1.5;
          ctx.setLineDash([5, 4]);
          ctx.beginPath(); ctx.moveTo(X(v), padT); ctx.lineTo(X(v), yBase); ctx.stroke();
          ctx.setLineDash([]);
        });
        ctx.fillStyle = C.text; ctx.font = '600 11px ui-monospace, monospace';
        ctx.textAlign = 'center';
        ctx.fillText('−bp', X(-bp), padT - 2);
        ctx.fillText('bp', X(bp), padT - 2);
      }
      if (state.mode === 'percentile') {
        [P_LO, P_HI].forEach(function (v) {
          ctx.strokeStyle = 'rgba(255,200,97,.8)'; ctx.lineWidth = 1.5;
          ctx.setLineDash([5, 4]);
          ctx.beginPath(); ctx.moveTo(X(v), padT); ctx.lineTo(X(v), yBase); ctx.stroke();
          ctx.setLineDash([]);
        });
        ctx.fillStyle = 'rgba(255,200,97,.9)'; ctx.font = '600 11px ui-monospace, monospace';
        ctx.textAlign = 'center'; ctx.fillText('clipped', X(P_HI), padT - 2);
      }

      /* quantization points */
      var yQ = yBase + 16;
      res.qp.forEach(function (v) {
        if (v < LO - 1 || v > HI + 1) return;
        var inDense = bp === null ? true : (v >= -bp - 1e-9 && v <= bp + 1e-9);
        ctx.fillStyle = inDense ? C.dense : C.outlier;
        ctx.beginPath(); ctx.arc(X(v), yQ, 3.1, 0, Math.PI * 2); ctx.fill();
      });

      ctx.fillStyle = C.muted; ctx.font = '11px ui-monospace, monospace';
      ctx.textAlign = 'left';  ctx.fillText(LO.toFixed(0), padL, H - 6);
      ctx.textAlign = 'right'; ctx.fillText(HI.toFixed(0), W - padR, H - 6);
      ctx.textAlign = 'center';
      ctx.fillText('quantization points', W / 2, H - 6);

      return res;
    }

    /* --- wiring ------------------------------------------------------ */
    var readMse   = document.getElementById('plq-mse');
    var readGain  = document.getElementById('plq-gain');
    var readLev   = document.getElementById('plq-levels');
    var bpRow     = document.getElementById('plq-bp-row');
    var bpOut     = document.getElementById('plq-bp-val');
    var bitsOut   = document.getElementById('plq-bits-val');

    function refresh() {
      var res = draw();
      var base = evaluate('minmax', state.bits, state.bpPct).mse;
      readMse.textContent = res.mse.toFixed(2);
      readLev.textContent = state.mode === 'ours'
        ? (Math.pow(2, state.bits) + ' + 2×' + Math.pow(2, state.bits - 1))
        : String(Math.pow(2, state.bits));
      var gain = 10 * Math.log10(base / res.mse);
      readGain.textContent = (gain >= 0 ? '+' : '') + gain.toFixed(2) + ' dB';
      readGain.className = gain > 0.05 ? 'win' : '';
      bpRow.style.display = state.mode === 'ours' ? '' : 'none';
      bpOut.textContent = (state.bpPct * 100).toFixed(1) + 'th';
      bitsOut.textContent = state.bits + '-bit';
    }

    document.querySelectorAll('#plq-mode button').forEach(function (btn) {
      btn.addEventListener('click', function () {
        state.mode = btn.dataset.mode;
        document.querySelectorAll('#plq-mode button').forEach(function (b) {
          b.setAttribute('aria-pressed', String(b === btn));
        });
        refresh();
      });
    });
    document.getElementById('plq-bits').addEventListener('input', function (e) {
      state.bits = +e.target.value; refresh();
    });
    document.getElementById('plq-bp').addEventListener('input', function (e) {
      state.bpPct = +e.target.value / 1000; refresh();
    });

    window.addEventListener('resize', refresh);
    refresh();
  }

  /* ================================================================
     Qualitative results explorer
     ================================================================ */
  var explorer = document.getElementById('explorer');
  if (explorer) initExplorer(explorer);

  function initExplorer(root) {
    var stage  = root.querySelector('.stage');
    var img    = stage.querySelector('img');
    var lens   = stage.querySelector('.lens');
    var badge  = stage.querySelector('.badge');
    var scene  = root.dataset.scene;
    var method = 'ours';

    var NAMES = {
      minmax: 'MinMax', percentile: 'Percentile', ptq4sr: 'PTQ4SR',
      adabm: 'AdaBM', ours: 'Ours', fp: 'Full-Precision'
    };

    function src(s, m) { return 'static/results/' + s + '/' + m + '.jpg'; }

    function preload(s) {
      Object.keys(NAMES).forEach(function (m) { new Image().src = src(s, m); });
    }

    function show() {
      img.src = src(scene, method);
      badge.textContent = NAMES[method] + (method === 'fp' ? '  ·  32-bit' : '  ·  W4A4');
      lens.style.backgroundImage = 'url("' + src(scene, method) + '")';
    }

    root.querySelectorAll('[data-method]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        method = btn.dataset.method;
        root.querySelectorAll('[data-method]').forEach(function (b) {
          b.setAttribute('aria-pressed', String(b === btn));
        });
        show();
      });
    });

    root.querySelectorAll('[data-scene]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        scene = btn.dataset.scene;
        root.querySelectorAll('[data-scene]').forEach(function (b) {
          b.setAttribute('aria-pressed', String(b === btn));
        });
        preload(scene); show();
      });
    });

    /* magnifier */
    var ZOOM = 3;
    stage.addEventListener('pointerenter', function () { stage.classList.add('zooming'); });
    stage.addEventListener('pointerleave', function () { stage.classList.remove('zooming'); });
    stage.addEventListener('pointermove', function (e) {
      var r = img.getBoundingClientRect();
      var x = e.clientX - r.left, y = e.clientY - r.top;
      if (x < 0 || y < 0 || x > r.width || y > r.height) {
        stage.classList.remove('zooming'); return;
      }
      stage.classList.add('zooming');
      var sr = stage.getBoundingClientRect();
      var size = lens.offsetWidth;
      lens.style.left = (e.clientX - sr.left - size / 2) + 'px';
      lens.style.top  = (e.clientY - sr.top  - size / 2) + 'px';
      lens.style.backgroundSize = (r.width * ZOOM) + 'px ' + (r.height * ZOOM) + 'px';
      lens.style.backgroundPosition =
        (-(x * ZOOM - size / 2)) + 'px ' + (-(y * ZOOM - size / 2)) + 'px';
    });

    preload(scene); show();
  }

  /* -------------------------------------------------- copy bibtex ------ */
  var copyBtn = document.querySelector('.copy');
  if (copyBtn) {
    copyBtn.addEventListener('click', function () {
      var text = document.querySelector('.bibtex pre').innerText;
      navigator.clipboard.writeText(text).then(function () {
        copyBtn.textContent = 'Copied';
        setTimeout(function () { copyBtn.textContent = 'Copy'; }, 1600);
      });
    });
  }
})();
