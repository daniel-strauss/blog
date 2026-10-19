(() => {
  const root = document.getElementById('bowl-demo');
  if (!root) return;
  const svg = root.querySelector('svg');
  const numerator = root.querySelector('#bowl-numerator');
  const denominator = root.querySelector('#bowl-denominator');
  const estimateButton = root.querySelector('#bowl-estimate');
  const progressPanel = root.querySelector('.bowl-estimate-progress');
  const progress = root.querySelector('#bowl-estimate-progress');
  const buttons = [...root.querySelectorAll('[data-bowl-mode]')];
  const drawButton = root.querySelector('#bowl-draw');
  const result = root.querySelector('#bowl-draw-result');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let mode = 'zero';
  let qWeights = readCurrentQ();
  let bowls = [];
  let generation = 0;
  let popAnimations = [];
  let estimate = null;
  let busy = null;
  const confidencePanel = root.querySelector('.bowl-confidence');
  const chart = {xs: [735, 805, 875], bottom: 520, height: 155};
  const percent = value => `${Number((100 * value).toFixed(1))}%`;
  const pickBowl = () => Math.floor(Math.random() * bowls.length);
  const pickDocument = bowl => Math.floor(Math.random() * bowl.contents.length);
  const stage = {x: 805, y: 180, scale: 1.13, documentY: 360};
  const gcd = (a, b) => b ? gcd(b, a % b) : a;
  const numeral = n => String(n).replace(/\d/g, digit => '⁰¹²³⁴⁵⁶⁷⁸⁹'[Number(digit)]);
  const label = index => `θ⁽${numeral(index + 1)}⁾`;
  const lerp = (a, b, t) => a + (b - a) * t;
  const ease = t => t < 0.5 ? 4 * t ** 3 : 1 - (-2 * t + 2) ** 3 / 2;
  const circle = (kind, x, y, index = 0, radius = 16) => `<g class="bowl-document" data-document-index="${index}"><circle class="bowl-circle bowl-circle-${kind.toLowerCase()}" cx="${x}" cy="${y}" r="${radius}"/><text class="bowl-circle-label" style="font-size:${radius < 16 ? 12 : 15}px" x="${x}" y="${y + 4}" text-anchor="middle">${kind}</text></g>`;

  function validWeights(weights) {
    return Array.isArray(weights) && weights.length === 3 && weights.every(w => Number.isInteger(w) && w >= 1 && w <= 6) && weights.reduce((a, b) => a + b, 0) <= 6;
  }
  function readCurrentQ() {
    try {
      const weights = JSON.parse(localStorage.getItem('icl-bowls-document-law-v1'));
      return validWeights(weights) ? weights : [1, 1, 1];
    } catch { return [1, 1, 1]; }
  }
  const probabilities = (weights = qWeights) => weights.map(w => w / weights.reduce((a, b) => a + b, 0));
  const qDescription = (weights = qWeights) => probabilities(weights).map((p, i) => `${'ABC'[i]} = ${percent(p)}`).join(' · ');
  function documentPosition(bowl, index) {
    if (bowl.contents.length <= 3) return {x: (index - (bowl.contents.length - 1) / 2) * 38, y: 20, radius: 16};
    const row = Math.floor(index / 3);
    const rowLength = Math.min(3, bowl.contents.length - row * 3);
    return {x: (index % 3 - (rowLength - 1) / 2) * 30, y: 3 + row * 28, radius: 12};
  }

  function contentsMarkup(bowl, index) {
    return `<text class="bowl-parameter" x="0" y="-36" text-anchor="middle">${label(index)}${mode === 'zero' ? ' = ψ' : ''}</text>
      <path class="bowl-vessel" d="M -77 -18 C -72 83 72 83 77 -18 Z"/>
      <ellipse class="bowl-inner-rim" cx="0" cy="-18" rx="77" ry="10"/>
      ${bowl.contents.map((document, j) => { const position = documentPosition(bowl, j); return circle(document, position.x, position.y, j, position.radius); }).join('')}`;
  }

  function configurations(p, q) {
    // Equal-probability copies preserve Q for every alpha = p/q:
    // W(q-p) mixed bowls and p*w_j singleton bowls for each document j.
    const weightSum = qWeights.reduce((a, b) => a + b, 0);
    const rawCounts = [weightSum * (q - p), ...qWeights.map(w => p * w)];
    const divisor = rawCounts.reduce(gcd);
    const [mixed, ...singles] = rawCounts.map(n => n / divisor);
    const mixedContents = qWeights.flatMap((w, i) => Array(w).fill('ABC'[i]));
    return [
      ...Array.from({length: mixed}, () => ({contents: mixedContents})),
      ...singles.flatMap((count, i) => Array.from({length: count}, () => ({contents: ['ABC'[i]]})))
    ];
  }

  function render(pop = false) {
    const columns = Math.min(bowls.length, bowls.length <= 6 ? 3 : bowls.length <= 15 ? 5 : 6);
    const scale = Math.min(1.04, 510 / (columns * 172));
    const rows = Math.ceil(bowls.length / columns);
    const pitch = 132 * scale;
    const outerBottom = Math.max(290, 154 + (rows - 1) * pitch + 85 * scale);
    const height = Math.max(estimate ? 640 : 475, outerBottom + 86);
    bowls.forEach((bowl, index) => {
      const row = Math.floor(index / columns);
      const rowLength = Math.min(columns, bowls.length - row * columns);
      bowl.x = 325 + (index % columns - (rowLength - 1) / 2) * (520 / columns);
      bowl.y = 154 + row * pitch;
      bowl.scale = scale;
    });
    svg.setAttribute('viewBox', `0 0 960 ${height}`);
    svg.innerHTML = `<title id="bowl-svg-title">Choose a numbered bowl, then draw one complete document</title>
      <desc id="bowl-svg-description">All ${bowls.length} bowl copies are shown and equally likely. A hand moves the selected bowl to the right, then draws one of its document circles uniformly. ${qDescription()}. Every representation induces this same Q.</desc>
      <rect class="bowl-stage-background" x="680" y="82" width="255" height="${height - 145}" rx="24"/>
      <text class="bowl-outer-label" x="325" y="54" text-anchor="middle">ψ · choose a fresh bowl for each document</text>
      <path class="bowl-outer" d="M 30 97 L 50 ${outerBottom - 20} Q 325 ${outerBottom + 45} 600 ${outerBottom - 20} L 620 97 Z"/>
      <ellipse class="bowl-rim" cx="325" cy="97" rx="295" ry="17"/>
      <g class="bowl-collection">${bowls.map((bowl, index) => `<g class="bowl-slot" data-bowl-index="${index}" style="--bowl-label-size: ${Math.max(19, 17 / bowl.scale)}px" data-documents="${bowl.contents.join('')}" transform="translate(${bowl.x} ${bowl.y}) scale(${bowl.scale})"><g class="bowl-inner">${contentsMarkup(bowl, index)}</g></g>`).join('')}</g>
      <text class="bowl-stage-label" x="805" y="114" text-anchor="middle">1. Pick a bowl</text>
      <ellipse class="bowl-stage-shadow" cx="805" cy="258" rx="89" ry="8"/>
      ${estimate ? histogramMarkup() : `<text class="bowl-stage-label" x="805" y="306" text-anchor="middle">2. Draw a document</text>
      <circle class="bowl-document-target" cx="805" cy="360" r="28"/>
      <text class="bowl-stage-hint" x="805" y="414" text-anchor="middle">Your sample lands here</text>`}
      <text class="bowl-output-label" x="480" y="${height - 18}" text-anchor="middle">Same Q: ${qDescription()}</text>
      <g class="bowl-animation-layer"></g>`;
    if (pop && !reducedMotion.matches) {
      popAnimations = [...svg.querySelectorAll('.bowl-inner')].map((node, index) => node.animate([
        {transform: 'translateY(25px) scale(0.05) rotate(-18deg)', opacity: 0},
        {transform: 'translateY(-8px) scale(1.12) rotate(6deg)', opacity: 1, offset: 0.65},
        {transform: 'translateY(3px) scale(0.96) rotate(-3deg)', offset: 0.84},
        {transform: 'translateY(0) scale(1) rotate(0)', opacity: 1}
      ], {duration: 530, delay: Math.min(index * 22, 330), fill: 'backwards', easing: 'ease-out'}));
    }
  }

  function update() {
    generation++;
    popAnimations.forEach(animation => animation.cancel());
    popAnimations = [];
    const denominatorValue = Number(denominator.value);
    numerator.max = String(denominatorValue);
    numerator.value = String(Math.min(Number(numerator.value), denominatorValue));
    const numeratorValue = Number(numerator.value);
    const [p, q] = mode === 'zero' ? [0, 1] : mode === 'full' ? [1, 1] : [numeratorValue, denominatorValue];
    const alpha = p / q;
    root.dataset.qWeights = JSON.stringify(qWeights);
    const entropy = probabilities().reduce((sum, p) => sum - p * Math.log(p), 0);
    root.querySelector('#bowl-entropy').textContent = `${entropy.toFixed(3)}/T`;
    bowls = configurations(p, q);
    buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.bowlMode === mode)));
    root.querySelector('.bowl-sliders').hidden = mode !== 'between';
    root.querySelector('#bowl-alpha-value').textContent = `α = ${numeratorValue}/${denominatorValue} · ${percent(numeratorValue / denominatorValue)}`;
    root.querySelector('#bowl-numerator-value').textContent = numeratorValue;
    root.querySelector('#bowl-denominator-value').textContent = denominatorValue;
    numerator.setAttribute('aria-valuetext', `${numeratorValue} out of ${denominatorValue}`);
    root.querySelector('#bowl-intra-value').textContent = percent(alpha);
    root.querySelector('#bowl-paper-total').textContent = `Paper’s estimation error = fixed meta term + ${percent(alpha)} of the canonical document uncertainty.`;
    root.querySelector('#bowl-noise-value').textContent = percent(1 - alpha);
    root.querySelector('.bowl-bar-intra').style.width = `${alpha * 100}%`;
    root.querySelector('.bowl-bar-noise').style.width = `${(1 - alpha) * 100}%`;
    root.querySelector('.bowl-bar').setAttribute('aria-label', `Of document uncertainty: ${percent(alpha)} intra-document contribution; ${percent(1 - alpha)} paper oracle baseline. Canonical estimation error and canonical irreducible error unchanged.`);
    root.querySelector('#bowl-explanation').textContent = mode === 'zero'
      ? 'θᵢ = ψ: one possible bowl, three document types. Repeated circles encode Q. Choosing the bowl adds no information.'
      : mode === 'full'
        ? 'θᵢ = Dᵢ: the parameter reveals the whole document, so the paper’s oracle baseline is zero. Uncertainty given Q is unchanged.'
        : `α = ${p}/${q}: ${bowls.length} equally likely bowls, every copy shown. Select one, then draw a circle uniformly from it.`;
    estimate = null;
    progressPanel.hidden = true;
    setBusy(null);
    result.textContent = 'Pick a bowl, then a complete document.';
    result.removeAttribute('data-document');
    render(true);
    updateConfidence();
  }

  function setBusy(kind) {
    busy = kind;
    drawButton.disabled = kind !== null;
    if (kind === 'sample') drawButton.setAttribute('aria-busy', 'true');
    else drawButton.removeAttribute('aria-busy');
    estimateButton.disabled = kind === 'sample';
    estimateButton.textContent = kind === 'estimate' ? 'Stop estimation' : 'Estimate distribution · 10,000 draws';
    estimateButton.setAttribute('aria-pressed', String(kind === 'estimate'));
  }

  function histogramMarkup() {
    return `<g class="bowl-histogram">
      <text class="bowl-stage-label" x="805" y="310" text-anchor="middle">Observed distribution</text>
      <text class="bowl-stage-hint" x="805" y="335" text-anchor="middle">Dashed marks: true Q</text>
      <line class="bowl-histogram-axis" x1="700" x2="910" y1="520" y2="520"/>
      ${['A', 'B', 'C'].map((document, i) => `<g data-histogram-document="${document}">
        <rect class="bowl-histogram-bar bowl-circle-${document.toLowerCase()}" x="${chart.xs[i] - 22}" y="520" width="44" height="0" rx="4"/>
        <line class="bowl-histogram-target" x1="${chart.xs[i] - 30}" x2="${chart.xs[i] + 30}" y1="417" y2="417"/>
        <path class="bowl-histogram-ci" d=""/>
        <text class="bowl-histogram-percent" x="${chart.xs[i]}" y="508" text-anchor="middle">—</text>
        <text class="bowl-parameter" x="${chart.xs[i]}" y="548" text-anchor="middle">${document}</text>
        <text class="bowl-histogram-count" x="${chart.xs[i]}" y="573" text-anchor="middle">0</text>
      </g>`).join('')}
      <text class="bowl-stage-hint" x="805" y="601" text-anchor="middle">Counts · 0 documents</text>
    </g>`;
  }

  function histogramGeometry(counts = estimate.counts, total = estimate.total) {
    const frequencies = counts.map(count => total ? count / total : 0);
    const intervals = counts.map(count => wilson(count, total));
    const truth = probabilities(estimate?.qWeights || qWeights);
    const maximum = Math.max(0.5, ...truth, ...intervals.map(interval => interval[1]));
    return {
      tops: frequencies.map(value => chart.bottom - chart.height * value / maximum),
      intervals: intervals.map(interval => interval.map(value => chart.bottom - chart.height * value / maximum)),
      targets: truth.map(p => chart.bottom - chart.height * p / maximum)
    };
  }

  function updateHistogram() {
    const geometry = histogramGeometry();
    const groups = [...svg.querySelectorAll('[data-histogram-document]')];
    groups.forEach((group, i) => {
      const bar = group.querySelector('rect');
      bar.setAttribute('y', geometry.tops[i]);
      bar.setAttribute('height', chart.bottom - geometry.tops[i]);
      const [lowY, highY] = geometry.intervals[i];
      const x = chart.xs[i];
      group.querySelector('.bowl-histogram-ci').setAttribute('d', estimate.total ? `M ${x} ${highY} V ${lowY} M ${x - 8} ${highY} H ${x + 8} M ${x - 8} ${lowY} H ${x + 8}` : '');
      group.querySelector('.bowl-histogram-percent').setAttribute('y', (estimate.total ? highY : geometry.tops[i]) - 12);
      group.querySelector('.bowl-histogram-percent').textContent = estimate.total ? percent(estimate.counts[i] / estimate.total) : '—';
      group.querySelector('.bowl-histogram-count').textContent = estimate.counts[i];
      group.querySelector('.bowl-histogram-target').setAttribute('y1', geometry.targets[i]);
      group.querySelector('.bowl-histogram-target').setAttribute('y2', geometry.targets[i]);
    });
    svg.querySelector('.bowl-histogram > .bowl-stage-hint:last-child').textContent = `Counts · ${estimate.total.toLocaleString()} documents`;
    progress.value = estimate.total;
    root.querySelector('#bowl-estimate-count').textContent = `${estimate.total.toLocaleString()} / 10,000`;
    updateConfidence();
  }

  function finishEstimate(stopped = false) {
    setBusy(null);
    updateConfidence();
    const summary = ['A', 'B', 'C'].map((document, i) => `${document}: ${estimate.counts[i]} (${estimate.total ? percent(estimate.counts[i] / estimate.total) : '—'})`).join(' · ');
    result.textContent = `${stopped ? 'Stopped' : 'Finished'} at ${estimate.total.toLocaleString()} draws. ${summary}`;
    result.removeAttribute('data-document');
    root.querySelector('#bowl-estimate-note').textContent = `${stopped ? 'Partial estimate' : '10,000 fresh bowl-and-document draws complete'}. Percentages use the completed draws; random fluctuations around Q are expected.`;
    svg.querySelector('.bowl-hand')?.setAttribute('visibility', 'hidden');
  }

  function stopEstimate() {
    generation++;
    // Remove the uncounted in-flight trial and show the completed observations.
    render();
    updateHistogram();
    finishEstimate(true);
  }

  // Two-sided 95% Wilson score interval for a binomial marginal.
  // Each document category is an indicator on the same iid document draws.
  function wilson(count, n) {
    if (!n) return [0, 1];
    const z = 1.959963984540054;
    const z2 = z * z;
    const p = count / n;
    const denominator = 1 + z2 / n;
    const center = (p + z2 / (2 * n)) / denominator;
    const half = z * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator;
    return [Math.max(0, center - half), Math.min(1, center + half)];
  }

  function updateConfidence() {
    confidencePanel.hidden = !estimate;
    if (!estimate) return;
    root.querySelector('.bowl-ci-values').innerHTML = ['A', 'B', 'C'].map((document, i) => {
      const interval = wilson(estimate.counts[i], estimate.total);
      return `<div><strong class="bowl-legend-dot bowl-legend-${document.toLowerCase()}">${document}</strong><span>${estimate.total ? `${percent(interval[0])}–${percent(interval[1])}` : '—'}</span></div>`;
    }).join('');
  }

  // One cancellable timeline: changing mode or alpha invalidates the old draw.
  function tween(run, duration, paint) {
    return new Promise(resolve => {
      const start = performance.now();
      function tick(now) {
        if (run !== generation) {resolve(false); return;}
        const t = Math.min(1, (now - start) / duration);
        paint(t);
        if (t < 1) requestAnimationFrame(tick);
        else resolve(true);
      }
      requestAnimationFrame(tick);
    });
  }
  const pose = (node, x, y, scale = 1, angle = 0) => node.setAttribute('transform', `translate(${x} ${y}) rotate(${angle}) scale(${scale})`);
  const handMarkup = `<g class="bowl-hand" aria-hidden="true">
    <path class="bowl-hand-palm" d="M -18 -53 L 15 -53 L 18 -27 Q 29 -34 37 -22 Q 44 -13 37 -7 L 23 -13 L 23 5 Q 23 20 13 19 L 7 15 Q 0 24 -8 16 Q -21 23 -25 9 L -29 -10 Q -36 -25 -28 -30 Q -23 -33 -17 -22 Z"/>
    <path class="bowl-hand-lines" d="M 7 15 L 7 -3 M -8 16 L -10 -3 M 23 -13 Q 15 -16 15 -24"/>
    <rect class="bowl-hand-cuff" x="-23" y="-62" width="43" height="17" rx="5"/>
  </g>`;

  async function sample() {
    const run = ++generation;
    popAnimations.forEach(animation => animation.cancel());
    estimate = null;
    progressPanel.hidden = true;
    render();
    setBusy('sample');
    updateConfidence();
    const index = pickBowl();
    const bowl = bowls[index];
    const source = svg.querySelector(`[data-bowl-index="${index}"]`);
    const layer = svg.querySelector('.bowl-animation-layer');
    layer.innerHTML = `<g class="bowl-carried bowl-selected">${contentsMarkup(bowl, index)}</g><g class="bowl-picked-document"></g><g class="bowl-sparkles"></g>${handMarkup}`;
    const carried = layer.querySelector('.bowl-carried');
    const hand = layer.querySelector('.bowl-hand');
    const picked = layer.querySelector('.bowl-picked-document');
    carried.style.opacity = '0';
    result.textContent = 'Reaching for a random bowl…';
    pose(hand, 900, 45, 1, -18);
    if (!reducedMotion.matches) {
      if (!await tween(run, 600, t => {
        const e = ease(t);
        pose(hand, lerp(900, bowl.x, e), lerp(45, bowl.y - 20 * bowl.scale, e) - 50 * Math.sin(Math.PI * t), 1, lerp(-18, 5, e));
      })) return;
    }
    source.style.opacity = '0.2';
    carried.style.opacity = '1';
    result.textContent = `Picked ${label(index)}. Moving it beside the big bowl…`;
    if (!reducedMotion.matches) {
      if (!await tween(run, 900, t => {
        const e = ease(t);
        const x = lerp(bowl.x, stage.x, e);
        const y = lerp(bowl.y, stage.y, e) - 100 * Math.sin(Math.PI * t);
        const scale = lerp(bowl.scale, stage.scale, e);
        const tilt = -12 * Math.sin(2 * Math.PI * t);
        pose(carried, x, y, scale, tilt);
        pose(hand, x, y - 20 * scale, 1, tilt);
      })) return;
    }
    pose(carried, stage.x, stage.y, stage.scale);
    // The second independent random choice happens after the bowl is selected.
    const documentIndex = pickDocument(bowl);
    const document = bowl.contents[documentIndex];
    const docPosition = documentPosition(bowl, documentIndex);
    const docX = stage.x + docPosition.x * stage.scale;
    const docY = stage.y + docPosition.y * stage.scale;
    result.textContent = `${label(index)} selected. Now picking a document from it…`;
    if (!reducedMotion.matches) {
      if (!await tween(run, 550, t => {
        const e = ease(t);
        pose(hand, lerp(stage.x, docX, e), lerp(stage.y - 20 * stage.scale, docY - 8, e) - 65 * Math.sin(Math.PI * t), 1, 10 * Math.sin(Math.PI * t));
      })) return;
    }
    carried.querySelector(`[data-document-index="${documentIndex}"]`).style.opacity = '0.2';
    picked.innerHTML = circle(document, 0, 0);
    if (!reducedMotion.matches) {
      if (!await tween(run, 700, t => {
        const e = ease(t);
        const x = lerp(docX, stage.x, e) + 38 * Math.sin(Math.PI * t);
        const y = lerp(docY, stage.documentY, e);
        pose(picked, x, y, lerp(stage.scale, 1.5, e), 20 * Math.sin(2 * Math.PI * t));
        pose(hand, x, y - 8, 1, 10 * Math.sin(Math.PI * t));
      })) return;
    }
    pose(picked, stage.x, stage.documentY, 1.5);
    result.textContent = `${label(index)} → document ${document}. Next sample starts with all bowls again.`;
    result.dataset.document = document;
    svg.querySelector('.bowl-stage-hint').textContent = `One complete document: ${document}`;
    if (!reducedMotion.matches) {
      const sparkles = layer.querySelector('.bowl-sparkles');
      sparkles.innerHTML = Array.from({length: 9}, (_, i) => `<rect x="-3" y="-5" width="6" height="10" rx="2" fill="${['#b25237', '#2c709b', '#7860a0', '#d79c27'][i % 4]}"/>`).join('');
      if (!await tween(run, 600, t => {
        const e = ease(t);
        pose(hand, lerp(stage.x, 930, e), lerp(stage.documentY - 8, 30, e), 1, 25 * e);
        hand.style.opacity = String(1 - e);
        [...sparkles.children].forEach((piece, i) => {
          const angle = i * Math.PI * 2 / 9;
          pose(piece, stage.x + Math.cos(angle) * 65 * t, stage.documentY + Math.sin(angle) * 65 * t + 28 * t * t, 1 - 0.5 * t, 240 * t + i * 30);
          piece.style.opacity = String(1 - t);
        });
      })) return;
    }
    hand.style.opacity = '0';
    if (run === generation) {
      setBusy(null);
    }
  }

  async function estimateDistribution() {
    const run = ++generation;
    popAnimations.forEach(animation => animation.cancel());
    const [p, q] = mode === 'zero' ? [0, 1] : mode === 'full' ? [1, 1] : [Number(numerator.value), Number(denominator.value)];
    estimate = {mode, p, q, qWeights: [...qWeights], counts: [0, 0, 0], total: 0};
    render();
    progressPanel.hidden = false;
    setBusy('estimate');
    updateConfidence();
    updateHistogram();
    result.removeAttribute('data-document');
    result.textContent = 'Estimating Q with 10,000 independent bowl-and-document draws…';
    root.querySelector('#bowl-estimate-note').textContent = 'Watch the first three and last three picks. Smooth acceleration and braking over five seconds; all 10,000 draws count.';
    const layer = svg.querySelector('.bowl-animation-layer');
    layer.innerHTML = `<g class="bowl-carried bowl-selected"></g><g class="bowl-picked-document"></g>${handMarkup}`;
    const carried = layer.querySelector('.bowl-carried');
    const picked = layer.querySelector('.bowl-picked-document');
    const hand = layer.querySelector('.bowl-hand');
    const slots = [...svg.querySelectorAll('.bowl-slot')];
    carried.style.opacity = '0';
    picked.style.opacity = '0';
    hand.style.opacity = reducedMotion.matches ? '0' : '1';
    let visual = null;
    function chooseDraw() {
      const index = pickBowl();
      const bowl = bowls[index];
      const documentIndex = pickDocument(bowl);
      const document = bowl.contents[documentIndex];
      return {index, bowl, documentIndex, document, column: 'ABC'.indexOf(document)};
    }
    function recordDraw(draw) {
      estimate.counts[draw.column]++;
      estimate.total++;

    }
    function present(draw) {
      if (visual) slots[visual.index].style.opacity = '1';
      visual = draw;
      carried.innerHTML = contentsMarkup(draw.bowl, draw.index);
      picked.innerHTML = circle(draw.document, 0, 0);
    }
    function paint(phase) {
      if (reducedMotion.matches || !visual) return;
      const {index, bowl, documentIndex, column} = visual;
      const position = documentPosition(bowl, documentIndex);
      const docX = stage.x + position.x * stage.scale;
      const docY = stage.y + position.y * stage.scale;
      carried.style.opacity = phase < 0.2 ? '0' : '1';
      picked.style.opacity = phase < 0.68 ? '0' : '1';
      slots[index].style.opacity = phase < 0.2 ? '1' : '0.2';
      carried.querySelector(`[data-document-index="${documentIndex}"]`).style.opacity = phase < 0.68 ? '1' : '0.2';
      if (phase < 0.2) {
        const e = ease(phase / 0.2);
        pose(hand, lerp(stage.x, bowl.x, e), lerp(65, bowl.y - 20 * bowl.scale, e) - 45 * Math.sin(Math.PI * e));
      } else if (phase < 0.5) {
        const e = ease((phase - 0.2) / 0.3);
        const x = lerp(bowl.x, stage.x, e);
        const y = lerp(bowl.y, stage.y, e) - 65 * Math.sin(Math.PI * e);
        const scale = lerp(bowl.scale, stage.scale, e);
        pose(carried, x, y, scale, -9 * Math.sin(2 * Math.PI * e));
        pose(hand, x, y - 20 * scale);
      } else if (phase < 0.68) {
        const e = ease((phase - 0.5) / 0.18);
        pose(carried, stage.x, stage.y, stage.scale);
        pose(hand, lerp(stage.x, docX, e), lerp(stage.y - 20 * stage.scale, docY - 8, e));
      } else {
        const e = ease(Math.min(1, (phase - 0.68) / 0.2));
        pose(carried, stage.x, stage.y, stage.scale);
        const x = lerp(docX, chart.xs[column], e);
        const y = lerp(docY, histogramGeometry().tops[column] - 9, e);
        pose(picked, x, y, 0.85);
        const returning = phase > 0.88 ? ease((phase - 0.88) / 0.12) : 0;
        picked.style.opacity = String(1 - returning);
        pose(hand, lerp(x, stage.x, returning), lerp(y - 8, 65, returning));
      }
    }
    // Continuous bell-shaped velocity: gentle at both ends, fastest midway.
    // The six boundary draws are real, individually displayed observations.
    const first = Array.from({length: 3}, chooseDraw);
    const last = Array.from({length: 3}, chooseDraw);
    let previousCycle = -1;
    let latest = null;
    if (!await tween(run, 5000, t => {
      const position = t === 1 ? 20 : 5 * t + 15 * (t - Math.sin(2 * Math.PI * t) / (2 * Math.PI));
      const cycle = Math.min(19, Math.floor(position));
      const phase = t === 1 ? 1 : position - cycle;
      const target = position <= 3 ? Math.floor(position)
        : position < 17 ? 3 + Math.floor(9994 * (position - 3) / 14)
        : 9997 + Math.floor(position - 17);
      while (estimate.total < Math.min(10000, target)) {
        latest = estimate.total < 3 ? first[estimate.total]
          : estimate.total >= 9997 ? last[estimate.total - 9997] : chooseDraw();
        recordDraw(latest);
      }
      if (cycle !== previousCycle) {
        present(cycle < 3 ? first[cycle] : cycle >= 17 ? last[cycle - 17] : latest);
        previousCycle = cycle;
        result.textContent = cycle < 3 || cycle >= 17
          ? `Draw ${cycle < 3 ? cycle + 1 : 9998 + cycle - 17} of 10,000: choose a bowl, then a document.`
          : 'Accelerating through the middle draws…';
      }
      paint(phase);
      updateHistogram();
    })) return;
    if (visual) slots[visual.index].style.opacity = '1';
    picked.style.opacity = '0';
    carried.style.opacity = '0';
    if (run === generation) finishEstimate();
  }

  buttons.forEach(button => button.addEventListener('click', () => {mode = button.dataset.bowlMode; update();}));
  numerator.addEventListener('input', update);
  denominator.addEventListener('input', update);
  estimateButton.addEventListener('click', () => busy === 'estimate' ? stopEstimate() : estimateDistribution());
  drawButton.addEventListener('click', sample);
  root.addEventListener('bowl-redraw-q', event => {
    if (!validWeights(event.detail?.weights)) return;
    qWeights = [...event.detail.weights];
    try { localStorage.setItem('icl-bowls-document-law-v1', JSON.stringify(qWeights)); } catch { /* The selected Q still works for this session. */ }
    update();
  });
  root.addEventListener('bowl-pause-for-universe', () => {
    if (busy === 'estimate') stopEstimate();
    else if (busy === 'sample') update();
  });
  root.querySelectorAll('.bowl-controls').forEach(element => {element.hidden = false;});
  update();
})();
