(() => {
  const root = document.getElementById('bowl-demo');
  const dialog = document.getElementById('universe-dialog');
  if (!root || !dialog) return;
  const trigger = root.querySelector('#bowl-redraw-q');
  const form = dialog.querySelector('form');
  const input = dialog.querySelector('#universe-nickname');
  const sequence = dialog.querySelector('.universe-sequence');
  const scene = dialog.querySelector('.universe-scene');
  const status = dialog.querySelector('.universe-status');
  const skipButton = dialog.querySelector('.universe-skip');
  const sound = dialog.querySelector('#universe-soundtrack');
  const soundButton = dialog.querySelector('.universe-sound');
  let soundEnabled = true;
  const effectsButton = dialog.querySelector('.universe-effects');
  let effectsEnabled = true;
  let effectsFrame = 0;
  let effectsFade = 1;
  let backgroundEye = null;
  let backgroundFrame = 0;
  let finishBackgroundGrowth = null;
  // 142 BPM; the bass drop is 4.74 seconds into our excerpt (source 5:39.74).
  const soundtrackBpm = 142;
  const dropTime = 4.74;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const SVG = 'http://www.w3.org/2000/svg';
  let generation = 0;
  let skip = false;
  let portal = null;
  let bodyOverflow = '';
  let gazeState = null;
  const ease = t => t < 0.5 ? 4 * t ** 3 : 1 - (-2 * t + 2) ** 3 / 2;
  const lerp = (a, b, t) => a + (b - a) * t;
  const pose = (node, x, y, scale = 1, angle = 0) => node.setAttribute('transform', `translate(${x} ${y}) rotate(${angle}) scale(${scale})`);

  function motion(run, duration, paint, skipAllowed = true, animated = true) {
    return new Promise(resolve => {
      const start = performance.now();
      function frame(now) {
        if (run !== generation || !dialog.open) { resolve(false); return; }
        const t = (animated && reducedMotion.matches) || (skip && skipAllowed) ? 1 : Math.min(1, (now - start) / duration);
        paint(t);
        if (t < 1) requestAnimationFrame(frame);
        else resolve(true);
      }
      requestAnimationFrame(frame);
    });
  }

  function close(completed = false) {
    generation++;
    cancelAnimationFrame(effectsFrame);
    cancelAnimationFrame(backgroundFrame);
    finishBackgroundGrowth?.(false);
    finishBackgroundGrowth = null;
    backgroundEye?.remove();
    backgroundEye = null;
    dialog.style.filter = '';
    sound.pause();
    sound.currentTime = 0;
    portal?.remove();
    portal = null;
    dialog.close();
    document.body.style.overflow = bodyOverflow;
    scene.replaceChildren();
    if (completed) {
      const heading = root.querySelector('#bowl-q-title');
      heading.tabIndex = -1;
      heading.focus({preventScroll: true});
      window.scrollTo({top: window.scrollY + root.getBoundingClientRect().top - 85, behavior: 'instant'});
    } else trigger.focus({preventScroll: true});
  }

  function cleanClone(node) {
    const clone = node.cloneNode(true);
    [clone, ...clone.querySelectorAll('*')].forEach(element => {
      element.removeAttribute('id');
      element.removeAttribute('aria-labelledby');
      element.removeAttribute('aria-describedby');
      element.removeAttribute('for');
      if (element.matches('button, input, a')) element.tabIndex = -1;
    });
    clone.querySelectorAll('script').forEach(script => script.remove());
    clone.inert = true;
    clone.setAttribute('aria-hidden', 'true');
    return clone;
  }

  function miniature() {
    const mini = document.createElement('div');
    mini.className = 'universe-miniature';
    mini.append(cleanClone(root.querySelector('.bowl-q-heading')));
    mini.append(cleanClone(root.querySelector('.bowl-modes')));
    mini.append(cleanClone(root.querySelector('.bowl-drawing')));
    return mini;
  }

  function newWeights() {
    const candidates = [[1, 1, 2], [1, 2, 1], [2, 1, 1], [1, 2, 2], [2, 1, 2], [2, 2, 1],
      [1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]];
    const previous = JSON.parse(root.dataset.qWeights || '[1,1,1]');
    const available = candidates.filter(weights => weights.join() !== previous.join());
    return [...available[Math.floor(Math.random() * available.length)]];
  }

  function playSoundtrack(restart = false) {
    if (!soundEnabled) return;
    if (restart) sound.currentTime = 0;
    sound.volume = 0.45;
    sound.play().then(() => {
      soundButton.textContent = 'Sound on';
      soundButton.setAttribute('aria-pressed', 'true');
    }).catch(() => {
      soundButton.textContent = 'Play sound';
      soundButton.setAttribute('aria-pressed', 'false');
    });
  }

  function startColorEffects() {
    cancelAnimationFrame(effectsFrame);
    effectsFade = 1;
    let previousActive = null;
    let previousLabel = '';
    function frame(now) {
      if (!dialog.open) return;
      const active = effectsEnabled && !reducedMotion.matches;
      if (backgroundEye) {
        if (backgroundEye.hasAttribute('hidden') === active) backgroundEye.toggleAttribute('hidden', !active);
        backgroundEye.style.opacity = String(0.55 * effectsFade);
      }
      const audioTime = sound.currentTime;
      // One complete opposite-color-and-back pulse per double-time beat.
      const phase = (audioTime - dropTime) * (2 * soundtrackBpm / 60);
      const inversion = active && !sound.paused && audioTime >= dropTime
        ? effectsFade * (0.5 + 0.5 * Math.cos(2 * Math.PI * phase)) : 0;
      dialog.style.filter = `invert(${inversion.toFixed(4)})`;
      const label = reducedMotion.matches ? 'Effects off (reduced motion)' : effectsEnabled ? 'Effects on' : 'Effects off';
      if (label !== previousLabel) { effectsButton.textContent = label; previousLabel = label; }
      if (active !== previousActive) { effectsButton.setAttribute('aria-pressed', String(active)); previousActive = active; }
      effectsFrame = requestAnimationFrame(frame);
    }
    effectsFrame = requestAnimationFrame(frame);
  }

  function scatteredEyes() {
    const positions = [[852, 191, 3.4, 0], [836, 233, 1.4, -14], [867, 234, 1.5, 19]];
    function candidate(scale) {
      // Sample the irregular silhouette, leaving the face and open mouth clear.
      for (;;) {
        const x = 778 + Math.random() * 151;
        const y = 184 + Math.random() * 326;
        const hood = y < 207 && Math.abs(x - 852) < 42;
        const hair = y >= 207 && y < 404 && (x < 815 || x > 891);
        const gown = y > 298 && Math.abs(x - 852) < 25 + (y - 298) * 0.25;
        if (hood || hair || gown) return [x, y, scale, Math.random() * 140 - 70];
      }
    }
    // Best-candidate scattering avoids a grid, while leaving room for large eyes.
    for (let i = 3; i < 100; i++) {
      const scale = 0.85 + Math.random() ** 1.3 * 0.95;
      let best, bestSpace = -Infinity;
      for (let attempt = 0; attempt < 36; attempt++) {
        const point = candidate(scale);
        const space = Math.min(...positions.map(([x, y, size]) => Math.hypot(point[0] - x, point[1] - y) / (size + scale)));
        if (space > bestSpace) { best = point; bestSpace = space; }
      }
      positions.push(best);
    }
    return positions.map(([x, y, scale, angle], index) => `<g class="creator-eye${index === 0 ? ' creator-eye-dominant' : ''}" data-angle="${angle}" data-wave-phase="${y * 0.024 + x * 0.013}" transform="translate(${x} ${y}) rotate(${angle}) scale(${scale})">
      <g class="creator-eye-drift">
        <path class="creator-eye-white" d="M -7 0 Q -1 -9 7 0 Q 1 9 -7 0 Z"/>
        <g clip-path="url(#universe-eye-aperture)">
          <g class="creator-pupil"><circle class="creator-iris" r="3.1"/><circle fill="#06080d" r="1.5"/></g>
          <ellipse fill="#fff" opacity=".8" cx="-1.1" cy="-1.2" rx="0.8" ry="0.5"/>
        </g>
      </g>
    </g>`).join('');
  }
  const creatorMarkup = () => `<g class="universe-creator" aria-label="A woman with one hundred moving eyes pushing the universe bowl">
    <path class="creator-wisp" d="M 822 182 Q 772 201 791 266 Q 808 312 774 359 Q 747 391 771 436 M 883 176 Q 935 208 918 274 Q 906 323 941 362 Q 964 396 939 452 M 792 371 Q 761 437 785 489 M 923 363 Q 953 434 931 485"/>
    <path class="creator-hair" d="M 790 233 Q 781 164 849 167 Q 916 156 920 234 L 938 412 Q 893 447 852 404 Q 803 447 770 412 Z"/>
    <path class="creator-robe" d="M 829 283 Q 852 297 876 283 Q 897 310 894 359 L 935 523 L 909 515 L 899 539 L 878 525 L 860 548 L 840 531 L 820 540 L 810 521 L 775 530 L 809 358 Q 807 310 829 283 Z"/>
    <path class="creator-fold" d="M 823 359 L 797 509 M 883 359 L 911 509"/>
    <path class="creator-robe" d="M 864 321 Q 809 321 744 337 L 745 363 Q 826 360 878 350 Z"/>
    <path class="creator-skin" d="M 745 338 L 725 331 Q 713 329 715 338 L 724 342 L 712 342 Q 704 345 712 349 L 726 353 Q 708 353 716 360 L 745 361 Z"/>
    <path class="creator-robe" d="M 832 308 Q 790 282 740 291 L 738 321 Q 789 323 829 346 Z"/>
    <path class="creator-skin" d="M 742 294 L 728 285 Q 719 279 716 287 L 721 295 L 710 293 Q 699 295 708 302 L 721 306 Q 704 305 708 313 L 739 319 Z"/>
    <path class="creator-skin" d="M 838 265 L 835 284 Q 850 301 869 284 L 865 264 Z"/>
    <path class="creator-skin" d="M 819 220 Q 817 194 850 193 Q 884 191 885 220 L 879 254 Q 869 275 852 279 Q 831 273 823 253 Z"/>
    <path class="creator-hair" d="M 812 231 Q 795 191 823 180 Q 858 165 889 192 L 895 230 Q 871 208 852 193 Q 841 219 812 231 Z"/>
    <path class="creator-feature" d="M 850 238 L 846 250 L 853 251"/>
    <ellipse class="creator-mouth" cx="853" cy="263" rx="5.5" ry="9"/>
    <path class="creator-veil" d="M 810 212 Q 815 263 802 291 M 894 209 Q 887 269 900 300 M 823 194 Q 849 180 881 199"/>
    ${scatteredEyes()}
  </g>`;

  // Shared saccades: fast target changes, short fixations, tiny drifting corrections.
  function gazeAt(now) {
    if (!gazeState) gazeState = {from: {x: 0, y: 0}, to: {x: 0, y: 0}, started: now, duration: 90, next: now};
    if (now >= gazeState.next) {
      const from = gazeState.to;
      let to;
      for (let attempt = 0; attempt < 12; attempt++) {
        const direction = Math.random() * Math.PI * 2;
        const radius = Math.random() < 0.15 ? 0.12 : 0.6 + Math.random() * 0.4;
        to = {x: radius * Math.cos(direction), y: radius * Math.sin(direction)};
        if (Math.hypot(to.x - from.x, to.y - from.y) > 0.55) break;
      }
      const duration = 75 + Math.random() * 45;
      gazeState = {from, to, started: now, duration, next: now + duration + 180 + Math.random() * 240};
    }
    const phase = Math.min(1, (now - gazeState.started) / gazeState.duration);
    const shift = phase * phase * phase * (phase * (phase * 6 - 15) + 10);
    return {
      x: lerp(gazeState.from.x, gazeState.to.x, shift) + 0.012 * Math.sin(now * 0.009),
      y: lerp(gazeState.from.y, gazeState.to.y, shift) + 0.009 * Math.cos(now * 0.011)
    };
  }

  function animateEyes(run, creator) {
    const pupils = [...creator.querySelectorAll('.creator-pupil')].map(node => ({
      node, eye: node.closest('.creator-eye-drift'), phase: Number(node.closest('.creator-eye').dataset.wavePhase),
      dominant: node.closest('.creator-eye').classList.contains('creator-eye-dominant'),
      angle: Number(node.closest('.creator-eye').dataset.angle) * Math.PI / 180,
      blinkOffset: Math.random() * 5, blinkPeriod: 3.7 + Math.random() * 2
    }));
    if (reducedMotion.matches) return;
    const start = performance.now();
    function frame(now) {
      if (run !== generation || !creator.isConnected || creator.style.opacity === '0') return;
      const t = (now - start) / 1000;
      const gaze = gazeAt(now);
      pupils.forEach(({node, eye, phase, blinkOffset, blinkPeriod, angle, dominant}) => {
        // Almost fourfold diameter change, with a visible traveling swell.
        const wave = Math.sin(t * 3.4 - phase);
        const size = dominant ? 1.1 + 0.05 * wave : 1.11 + 0.69 * wave;
        const blinkPhase = (t + blinkOffset) % blinkPeriod;
        const blink = blinkPhase < 0.16 ? 1 - 0.92 * Math.sin(Math.PI * blinkPhase / 0.16) ** 2 : 1;
        const opening = dominant ? 1 : 0.94 + 0.14 * Math.sin(t * 3.4 - phase + 0.4);
        const rotation = angle;
        // Undo each eye's individual rotation/stretch so all gazes agree on screen.
        const x = 2.35 * (Math.cos(rotation) * gaze.x + Math.sin(rotation) * gaze.y);
        const y = 2.35 * (-Math.sin(rotation) * gaze.x + Math.cos(rotation) * gaze.y) / opening;
        node.setAttribute('transform', `translate(${x.toFixed(3)} ${y.toFixed(3)})`);
        eye.setAttribute('transform', `scale(${size.toFixed(3)} ${(size * opening * (dominant ? 1 : blink)).toFixed(3)})`);
      });
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  function growBackgroundEye(run, source) {
    if (reducedMotion.matches || !effectsEnabled) return Promise.resolve(true);
    const origin = source.getBoundingClientRect();
    const sourceShape = source.querySelector('.creator-eye-drift');
    const sourceMatrix = sourceShape.getScreenCTM();
    const bounds = dialog.getBoundingClientRect();
    const startX = sourceMatrix ? sourceMatrix.e - bounds.left : origin.left + origin.width / 2 - bounds.left;
    const startY = sourceMatrix ? sourceMatrix.f - bounds.top : origin.top + origin.height / 2 - bounds.top;
    const startScale = Math.max(0.2, sourceMatrix ? Math.hypot(sourceMatrix.a, sourceMatrix.b) : origin.width / 14);
    backgroundEye = document.createElementNS(SVG, 'svg');
    backgroundEye.classList.add('universe-eye-backdrop');
    backgroundEye.setAttribute('aria-hidden', 'true');
    backgroundEye.innerHTML = `<g class="universe-growing-eye">
      <path class="universe-giant-eye-white" d="M -7 0 Q -1 -9 7 0 Q 1 9 -7 0 Z"/>
      <circle class="universe-giant-iris" r="2.9"/>
    </g><g class="universe-eye-rings"></g><g class="universe-giant-pupil"><ellipse rx="1.25" ry="2" fill="#060311"/><circle cx="0.65" cy="-0.8" r="0.45" fill="#fff" opacity=".85"/></g>`;
    dialog.prepend(backgroundEye);
    source.style.opacity = '0';
    const layer = backgroundEye;
    const eye = layer.querySelector('.universe-growing-eye');
    const pupil = layer.querySelector('.universe-giant-pupil');
    const iris = layer.querySelector('.universe-giant-iris');
    let previousWidth = 0, previousHeight = 0;
    const ringLayer = layer.querySelector('.universe-eye-rings');
    const rings = [];
    const started = performance.now();
    let lastRing = -Infinity;
    let settled = false;
    return new Promise(resolve => {
      finishBackgroundGrowth = resolve;
      function frame(now) {
        if (run !== generation || !dialog.open || !layer.isConnected) { if (!settled) resolve(false); return; }
        const elapsed = (now - started) / 1000;
        const t = skip || reducedMotion.matches || !effectsEnabled ? 1 : Math.min(1, elapsed / 6.5);
        const width = dialog.clientWidth;
        const height = dialog.clientHeight;
        if (width !== previousWidth || height !== previousHeight) {
          layer.setAttribute('viewBox', `0 0 ${width} ${height}`);
          previousWidth = width; previousHeight = height;
        }
        const targetScale = Math.max(width / 14, height / 9) * 1.12;
        const scale = lerp(startScale, targetScale, t ** 1.5) * Math.exp(Math.max(0, elapsed - 6.5) * 0.035);
        const centerX = lerp(startX, width / 2, ease(t));
        const centerY = lerp(startY, height / 2, ease(t));
        const gaze = gazeAt(now);
        const lookX = 0.2 * gaze.x;
        const lookY = 0.2 * gaze.y;
        const pupilX = centerX + lookX * scale;
        const pupilY = centerY + lookY * scale;
        pose(eye, centerX, centerY, scale);
        pose(pupil, pupilX, pupilY, scale);
        iris.setAttribute('cx', lookX);
        iris.setAttribute('cy', lookY);
        pose(ringLayer, pupilX, pupilY);
        if (elapsed - lastRing > 0.24 && effectsEnabled && !reducedMotion.matches) {
          const node = document.createElementNS(SVG, 'circle');
          node.setAttribute('fill', 'none');
          node.setAttribute('stroke', `hsl(${Math.random() * 360} 100% 60%)`);
          node.setAttribute('stroke-width', 3 + Math.random() * 7);
          ringLayer.append(node);
          rings.push({node, born: elapsed});
          lastRing = elapsed;
        }
        const reach = Math.hypot(width, height);
        for (let i = rings.length - 1; i >= 0; i--) {
          const ring = rings[i];
          const age = (elapsed - ring.born) / 3;
          if (age >= 1) { ring.node.remove(); rings.splice(i, 1); continue; }
          ring.node.setAttribute('r', age * reach);
          ring.node.setAttribute('opacity', 0.9 * (1 - age));
        }
        if (t === 1 && !settled) { settled = true; finishBackgroundGrowth = null; resolve(true); }
        backgroundFrame = requestAnimationFrame(frame);
      }
      backgroundFrame = requestAnimationFrame(frame);
    });
  }

  function buildScene() {
    scene.innerHTML = `<defs>
      <clipPath id="universe-eye-aperture"><path d="M -7 0 Q -1 -9 7 0 Q 1 9 -7 0 Z"/></clipPath>
      <radialGradient id="universe-iris-glow"><stop offset="0" stop-color="#f2ff6e"/><stop offset=".5" stop-color="#b6ff28"/><stop offset="1" stop-color="#08755f"/></radialGradient>
      <radialGradient id="universe-sclera-glow" cx="42%" cy="35%" r="75%"><stop offset="0" stop-color="#fffef1"/><stop offset=".65" stop-color="#f6ffcf"/><stop offset="1" stop-color="#7cbba0"/></radialGradient>
      <radialGradient id="universe-glow"><stop offset="0" stop-color="#70a98f" stop-opacity=".2"/><stop offset="1" stop-color="#070b12" stop-opacity="0"/></radialGradient>
      </defs>
      <ellipse cx="390" cy="340" rx="450" ry="310" fill="url(#universe-glow)"/>
      ${Array.from({length: 75}, () => `<circle cx="${Math.random() * 1000}" cy="${Math.random() * 650}" r="${0.3 + Math.random() * 1.2}" fill="#d7dfeb" opacity="${0.1 + Math.random() * 0.4}"/>`).join('')}
      <g class="universe-bowl-assembly">
      <text class="universe-scene-label" x="390" y="115" text-anchor="middle">A bowl of possible universes</text>
      <path class="universe-bowl" d="M 55 175 L 80 525 Q 390 650 700 525 L 725 175 Z"/>
      <ellipse class="universe-rim" cx="390" cy="175" rx="335" ry="22"/>
      <g class="universe-cards"></g>
      </g>
      ${creatorMarkup()}
      <g class="universe-sampling-stage">
      <ellipse cx="860" cy="440" rx="85" ry="10" fill="#172522"/>
      <text class="universe-scene-label universe-observation" x="860" y="480" text-anchor="middle">Looking for you…</text>
      <g class="universe-picked"></g>
      <g class="universe-hand" aria-hidden="true">
        <path class="bowl-hand-palm" d="M -18 -53 L 15 -53 L 18 -27 Q 29 -34 37 -22 Q 44 -13 37 -7 L 23 -13 L 23 5 Q 23 20 13 19 L 7 15 Q 0 24 -8 16 Q -21 23 -25 9 L -29 -10 Q -36 -25 -28 -30 Q -23 -33 -17 -22 Z"/>
        <path class="bowl-hand-lines" d="M 7 15 L 7 -3 M -8 16 L -10 -3 M 23 -13 Q 15 -16 15 -24"/>
        <rect class="bowl-hand-cuff" x="-23" y="-62" width="43" height="17" rx="5"/>
      </g>
      </g>`;
    const positions = Array.from({length: 30}, (_, i) => i);
    // Six bright universes among thirty; sampling is with replacement.
    for (let i = positions.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [positions[i], positions[j]] = [positions[j], positions[i]];
    }
    const bright = new Set(positions.slice(0, 6));
    const collection = scene.querySelector('.universe-cards');
    const template = miniature();
    const cards = positions.map((_, index) => {
      const x = 135 + (index % 6) * 100;
      const y = 235 + Math.floor(index / 6) * 66;
      const node = document.createElementNS(SVG, 'g');
      node.classList.add('universe-card');
      node.dataset.bright = String(bright.has(index));
      pose(node, x, y, 1, (index % 3 - 1) * 3);
      const foreign = document.createElementNS(SVG, 'foreignObject');
      foreign.setAttribute('x', '-42');
      foreign.setAttribute('y', '-30');
      foreign.setAttribute('width', '84');
      foreign.setAttribute('height', '62');
      const screen = document.createElement('div');
      screen.className = `universe-screen ${bright.has(index) ? 'is-bright' : 'is-dark'}`;
      screen.append(template.cloneNode(true));
      foreign.append(screen);
      node.append(foreign);
      collection.append(node);
      return {node, x, y, bright: bright.has(index)};
    });
    return cards;
  }

  async function zoomInto(run, selected) {
    root.dispatchEvent(new CustomEvent('bowl-redraw-q', {detail: {weights: newWeights()}}));
    // Align the live Q view under the portal before the final camera move.
    window.scrollTo({top: window.scrollY + root.getBoundingClientRect().top - 85, behavior: 'instant'});
    const target = root.getBoundingClientRect();
    const origin = selected.getBoundingClientRect();
    portal = cleanClone(root);
    portal.classList.add('universe-portal');
    portal.style.width = `${target.width}px`;
    portal.style.left = `${origin.left}px`;
    portal.style.top = `${origin.top}px`;
    portal.style.transform = `scale(${origin.width / target.width})`;
    dialog.append(portal);
    selected.style.opacity = '0';
    const startScale = origin.width / target.width;
    if (!await motion(run, 1400, t => {
      const e = ease(t);
      portal.style.left = `${lerp(origin.left, target.left, e)}px`;
      portal.style.top = `${lerp(origin.top, target.top, e)}px`;
      portal.style.transform = `scale(${lerp(startScale, 1, e)})`;
      sequence.style.opacity = String(1 - e);
      effectsFade = 1 - e;
      sequence.style.transform = `scale(${1 + e * 0.3})`;
      const darkness = 1 - Math.max(0, (e - 0.65) / 0.35);
      dialog.style.backgroundColor = `rgba(2, 4, 8, ${darkness})`;
      dialog.style.setProperty('--universe-backdrop-alpha', darkness);
    }, false)) return;
    if (!await motion(run, 220, t => { portal.style.opacity = String(1 - t); }, false)) return;
    close(true);
  }

  async function generate(nickname) {
    gazeState = null;
    const occurrence = `${nickname} uses this online tool and redraws Q`;
    const run = ++generation;
    const story = dialog.querySelector('.universe-story');
    story.style.transform = '';
    skip = false;
    form.hidden = true;
    sequence.hidden = false;
    sequence.style.opacity = '1';
    sequence.style.transform = '';
    dialog.querySelector('.universe-occurrence').textContent = `until occurrence of “${occurrence}”.`;
    status.hidden = true;
    scene.hidden = true;
    skipButton.hidden = false;
    // Let both equally sized lines stand alone before the bowl appears.
    if (!await motion(run, 3500, () => {}, true, false)) return;
    const previousTop = story.getBoundingClientRect().top;
    const cards = buildScene();
    scene.hidden = false;
    status.hidden = false;
    status.textContent = 'A little help getting things started.';
    scene.style.opacity = '0';
    const hand = scene.querySelector('.universe-hand');
    const picked = scene.querySelector('.universe-picked');
    const assembly = scene.querySelector('.universe-bowl-assembly');
    const creator = scene.querySelector('.universe-creator');
    const samplingStage = scene.querySelector('.universe-sampling-stage');
    hand.style.opacity = '0';
    samplingStage.style.opacity = '0';
    pose(assembly, 1050, 0);
    creator.setAttribute('transform', 'translate(1050 0)');
    const storyOffset = previousTop - story.getBoundingClientRect().top;
    if (!await motion(run, 650, t => {
      const e = ease(t);
      story.style.transform = `translateY(${storyOffset * (1 - e)}px)`;
      scene.style.opacity = String(e);
    })) return;
    story.style.transform = '';
    // Her hands stay on the bowl as she pushes it into view.
    animateEyes(run, creator);
    if (!await motion(run, 4000, t => {
      const e = t * t * (3 - 2 * t);
      const x = 1050 * (1 - e);
      pose(assembly, x, 0);
      creator.setAttribute('transform', `translate(${x} ${-3 * Math.sin(t * Math.PI * 8)}) rotate(${-3 * Math.sin(Math.PI * t)} 850 350)`);
    })) return;
    pose(assembly, 0, 0);
    creator.setAttribute('transform', 'translate(0 0)');
    status.textContent = 'That should do it.';
    if (!await motion(run, 3200, () => {})) return;
    // Recede behind the bowl instead of exiting sideways.
    scene.insertBefore(creator, assembly);
    if (!await motion(run, 3200, t => {
      const e = ease(t);
      const scale = lerp(1, 0.28, e);
      creator.setAttribute('transform', `translate(${lerp(850, 700, e)} ${lerp(350, 125, e)}) scale(${scale}) translate(-850 -350)`);
      creator.style.opacity = String(lerp(1, 0.5, e));
      samplingStage.style.opacity = String(e);
    })) return;
    const eyeReady = growBackgroundEye(run, creator.querySelector('.creator-eye-dominant'));
    let attempts = 0;
    while (run === generation) {
      // Stage three misses before the remaining random universe search.
      const candidates = attempts < 3 ? cards.filter(card => !card.bright)
        : skip || reducedMotion.matches ? cards.filter(card => card.bright) : cards;
      const card = candidates[Math.floor(Math.random() * candidates.length)];
      attempts++;
      picked.replaceChildren(card.node.firstChild.cloneNode(true));
      picked.style.opacity = '0';
      hand.style.opacity = '1';
      status.textContent = `Universe ${attempts.toLocaleString()}…`;
      if (!await motion(run, attempts === 1 ? 450 : 220, t => {
        const e = ease(t);
        pose(hand, lerp(860, card.x, e), lerp(110, card.y - 25, e) - 55 * Math.sin(Math.PI * e), 1.3, -10 * Math.sin(Math.PI * e));
      })) return;
      card.node.style.opacity = '0.2';
      picked.style.opacity = '1';
      if (!await motion(run, attempts === 1 ? 650 : 380, t => {
        const e = ease(t);
        const x = lerp(card.x, 860, e);
        const y = lerp(card.y, 350, e) - 90 * Math.sin(Math.PI * e);
        const scale = lerp(1, 1.6, e);
        pose(picked, x, y, scale, -12 * Math.sin(2 * Math.PI * e));
        pose(hand, x, y - 25 * scale, 1.3, -8 * Math.sin(Math.PI * e));
      })) return;
      if (card.bright) {
        status.textContent = `There you are. “${occurrence}”.`;
        scene.querySelector('.universe-observation').textContent = 'Occurrence found.';
        skipButton.hidden = true;
        if (!await motion(run, 500, t => { hand.style.opacity = String(1 - t); })) return;
        if (!await eyeReady || run !== generation) return;
        await zoomInto(run, picked);
        return;
      }
      status.textContent = `No “${occurrence}” in that one. Again.`;
      scene.querySelector('.universe-observation').textContent = 'No Valid Occurence';
      if (!await motion(run, 220, t => {
        picked.style.opacity = String(1 - t);
        pose(picked, 860, 350 + 50 * t, 1.6, 14 * t);
      })) return;
      card.node.style.opacity = '1';
    }
  }

  trigger.addEventListener('click', () => {
    root.dispatchEvent(new CustomEvent('bowl-pause-for-universe'));
    generation++;
    skip = false;
    form.hidden = false;
    sequence.hidden = true;
    dialog.style.backgroundColor = '';
    dialog.style.setProperty('--universe-backdrop-alpha', '1');
    bodyOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    dialog.showModal();
    startColorEffects();
    playSoundtrack(true);
    input.focus();
  });
  form.addEventListener('submit', event => {
    event.preventDefault();
    const nickname = input.value.trim();
    if (!nickname) { input.setCustomValidity('Enter your nickname.'); input.reportValidity(); return; }
    generate(nickname);
  });
  effectsButton.addEventListener('click', () => {
    effectsEnabled = !effectsEnabled;
  });
  soundButton.addEventListener('click', () => {
    if (!sound.paused) {
      soundEnabled = false;
      sound.pause();
      soundButton.textContent = 'Sound off';
      soundButton.setAttribute('aria-pressed', 'false');
    } else {
      soundEnabled = true;
      playSoundtrack();
    }
  });
  sound.addEventListener('ended', () => {
    soundButton.textContent = 'Replay sound';
    soundButton.setAttribute('aria-pressed', 'false');
  });
  input.addEventListener('input', () => input.setCustomValidity(''));
  skipButton.addEventListener('click', () => { skip = true; });
  dialog.querySelector('.universe-close').addEventListener('click', () => close());
  dialog.addEventListener('cancel', event => { event.preventDefault(); close(); });
})();
