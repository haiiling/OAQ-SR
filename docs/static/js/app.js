/* ============================================================
   OAQ-SR — project page interactions
   ============================================================ */
(function () {
  'use strict';

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
