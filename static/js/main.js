/**
 * Emotion-Aware Sarcasm Detection System
 * Main JavaScript - Handles API calls and UI updates
 */

// ---- DOM Elements ----
const inputText = document.getElementById('inputText');
const charCount = document.getElementById('charCount');
const analyseBtn = document.getElementById('analyseBtn');
const loading = document.getElementById('loading');
const errorDisplay = document.getElementById('errorDisplay');
const errorMessage = document.getElementById('errorMessage');
const results = document.getElementById('results');

// ---- Character Counter ----
inputText.addEventListener('input', () => {
    const len = inputText.value.length;
    charCount.textContent = `${len} / 500`;
    charCount.style.color = len > 450 ? '#ef476f' : '';
});

// Handle Enter key (Ctrl+Enter to submit)
inputText.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        analyseText();
    }
});

// ---- Set Example Text ----
function setExample(text) {
    inputText.value = text;
    charCount.textContent = `${text.length} / 500`;
    analyseText();
}

// ---- Emotion Color Map ----
const EMOTION_COLORS = {
    joy: '#FFD93D',
    anger: '#FF6B6B',
    sadness: '#74B9FF',
    fear: '#A29BFE',
    surprise: '#FD79A8',
    disgust: '#00B894',
    neutral: '#B2BEC3',
};

// ---- Analyse Text ----
async function analyseText() {
    const text = inputText.value.trim();
    if (!text) {
        showError('Please enter some text to analyse.');
        return;
    }

    if (text.length < 3) {
        showError('Text is too short. Please enter at least 3 characters.');
        return;
    }

    // Show loading, hide previous results
    loading.classList.remove('hidden');
    results.classList.add('hidden');
    errorDisplay.classList.add('hidden');
    analyseBtn.disabled = true;

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });

        const data = await response.json();

        if (!data.success) {
            showError(data.error || 'An unknown error occurred.');
            return;
        }

        displayResults(data);

    } catch (err) {
        showError('Failed to connect to the server. Please ensure it is running.');
        console.error(err);
    } finally {
        loading.classList.add('hidden');
        analyseBtn.disabled = false;
    }
}

// ---- Display Results ----
function displayResults(data) {
    // ---- Emotion ----
    const emotionLabel = document.getElementById('emotionLabel');
    emotionLabel.textContent = data.emotion.label;
    emotionLabel.style.background = EMOTION_COLORS[data.emotion.label] || EMOTION_COLORS.neutral;

    document.getElementById('emotionTime').textContent = `${data.emotion.inference_time_ms} ms`;

    // Emotion bars
    const barsContainer = document.getElementById('emotionBars');
    barsContainer.innerHTML = '';

    // Sort emotions by score descending
    const sortedEmotions = Object.entries(data.emotion.scores)
        .sort((a, b) => b[1] - a[1]);

    for (const [emotion, score] of sortedEmotions) {
        const percentage = (score * 100).toFixed(1);
        const color = EMOTION_COLORS[emotion] || '#ccc';

        const row = document.createElement('div');
        row.className = 'emotion-bar-row';
        row.innerHTML = `
            <span class="emotion-bar-label">${emotion}</span>
            <div class="emotion-bar-track">
                <div class="emotion-bar-fill" style="width: ${percentage}%; background: ${color};"></div>
            </div>
            <span class="emotion-bar-value">${percentage}%</span>
        `;
        barsContainer.appendChild(row);
    }

    // ---- Baseline Model ----
    updateModelCard(
        'baseline',
        data.baseline.label,
        data.baseline.confidence,
        data.baseline.probabilities,
        data.baseline.inference_time_ms
    );

    // ---- Emotion-Aware Model ----
    updateModelCard(
        'emotionAware',
        data.emotion_aware.label,
        data.emotion_aware.confidence,
        data.emotion_aware.probabilities,
        data.emotion_aware.inference_time_ms
    );

    // ---- Limitation Notice ----
    document.getElementById('limitationNotice').textContent = data.limitation_notice;

    // Show results
    results.classList.remove('hidden');

    // Scroll to results
    document.querySelector('.emotion-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---- Update Model Card ----
function updateModelCard(prefix, label, confidence, probabilities, timeMs) {
    const isSarcastic = label === 'sarcastic';
    const cssClass = isSarcastic ? 'sarcastic' : 'non-sarcastic';

    // Card border
    const card = document.getElementById(`${prefix}Card`);
    card.classList.remove('sarcastic', 'non-sarcastic');
    card.classList.add(cssClass);

    // Label
    const labelEl = document.getElementById(`${prefix}Label`);
    labelEl.textContent = label.toUpperCase();
    labelEl.className = `prediction-label ${cssClass}`;

    // Confidence bar
    const confBar = document.getElementById(`${prefix}ConfidenceBar`);
    confBar.style.width = `${(confidence * 100).toFixed(1)}%`;
    confBar.className = `confidence-bar ${cssClass}`;

    // Confidence text
    document.getElementById(`${prefix}Confidence`).textContent =
        `Confidence: ${(confidence * 100).toFixed(1)}%`;

    // Probabilities
    const probsEl = document.getElementById(`${prefix}Probs`);
    probsEl.innerHTML = `
        <div style="display:flex;justify-content:space-between;">
            <span>Non-sarcastic: ${(probabilities['non-sarcastic'] * 100).toFixed(1)}%</span>
            <span>Sarcastic: ${(probabilities['sarcastic'] * 100).toFixed(1)}%</span>
        </div>
    `;

    // Time
    document.getElementById(`${prefix}Time`).textContent = `Inference: ${timeMs} ms`;
}

// ---- Show Error ----
function showError(message) {
    loading.classList.add('hidden');
    results.classList.add('hidden');
    errorDisplay.classList.remove('hidden');
    errorMessage.textContent = message;
    analyseBtn.disabled = false;
}

// ---- Evaluation Modal ----
async function showEvaluation() {
    const modal = document.getElementById('evalModal');
    const content = document.getElementById('evalContent');
    modal.classList.remove('hidden');

    content.innerHTML = '<div class="spinner"></div><p>Loading evaluation results...</p>';

    try {
        const response = await fetch('/evaluate');
        const data = await response.json();

        if (!data.success) {
            content.innerHTML = `<p style="color: var(--sarcastic);">⚠️ ${data.error}</p>`;
            return;
        }

        content.innerHTML = buildEvaluationHTML(data);
    } catch (err) {
        content.innerHTML = '<p style="color: var(--sarcastic);">⚠️ Failed to load evaluation results.</p>';
    }
}

// ---- Build Evaluation HTML ----
function buildEvaluationHTML(data) {
    const comp = data.comparison;
    const bl = comp.baseline;
    const ea = comp.emotion_aware;

    let html = '<div class="eval-metrics-grid">';

    // Baseline metrics
    html += `<div class="eval-model-section">
        <h4>📄 Text-Only Baseline</h4>
        ${buildMetricsHTML(bl)}
        ${buildConfusionMatrix(bl.confusion_matrix)}
    </div>`;

    // Emotion-aware metrics
    html += `<div class="eval-model-section">
        <h4>🧠 Emotion-Aware Model</h4>
        ${buildMetricsHTML(ea)}
        ${buildConfusionMatrix(ea.confusion_matrix)}
    </div>`;

    html += '</div>';

    // Improvement
    html += '<div class="eval-improvement"><h4>📈 Improvement Analysis</h4>';
    for (const [metric, info] of Object.entries(comp.improvement)) {
        const arrow = info.improved ? '↑' : info.absolute < 0 ? '↓' : '→';
        const color = info.improved ? 'var(--non-sarcastic)' : info.absolute < 0 ? 'var(--sarcastic)' : 'var(--text-muted)';
        html += `<div class="eval-metric-row">
            <span class="label">${formatMetricName(metric)}</span>
            <span class="value" style="color:${color}">${info.absolute >= 0 ? '+' : ''}${(info.absolute * 100).toFixed(2)}% ${arrow}</span>
        </div>`;
    }
    html += '</div>';

    // Verdict
    html += `<div class="eval-verdict">${comp.verdict}</div>`;

    // Dataset info
    if (data.dataset) {
        html += `<div style="margin-top:16px; font-size:0.82rem; color:var(--text-secondary);">
            <strong>Dataset:</strong> ${data.dataset.name || 'N/A'} |
            <strong>Source:</strong> ${data.dataset.source || 'N/A'} |
            <strong>Seed:</strong> ${data.dataset.random_seed || 'N/A'}
        </div>`;
    }

    html += `<p style="margin-top:12px; font-size:0.82rem; color:var(--text-muted); font-style:italic;">
        ${comp.note || ''}
    </p>`;

    return html;
}

function buildMetricsHTML(metrics) {
    const rows = [
        ['Accuracy', metrics.accuracy],
        ['Precision', metrics.precision],
        ['Recall', metrics.recall],
        ['F1-Score', metrics.f1_score],
        ['Macro-F1', metrics.macro_f1],
        ['Latency/sample', `${metrics.latency_per_sample_ms} ms`],
    ];

    return rows.map(([label, value]) => `
        <div class="eval-metric-row">
            <span class="label">${label}</span>
            <span class="value">${typeof value === 'number' ? (value * 100).toFixed(2) + '%' : value}</span>
        </div>
    `).join('');
}

function buildConfusionMatrix(cm) {
    if (!cm || cm.length < 2) return '';
    return `
        <div style="margin-top:10px;">
            <span style="font-size:0.8rem; color:var(--text-secondary); font-weight:500;">Confusion Matrix:</span>
            <div class="confusion-matrix">
                <div class="cm-cell header"></div>
                <div class="cm-cell header">Pred 0</div>
                <div class="cm-cell header">Pred 1</div>
                <div class="cm-cell header">True 0</div>
                <div class="cm-cell tn">${cm[0][0]}</div>
                <div class="cm-cell fp">${cm[0][1]}</div>
                <div class="cm-cell header">True 1</div>
                <div class="cm-cell fn">${cm[1][0]}</div>
                <div class="cm-cell tp">${cm[1][1]}</div>
            </div>
        </div>
    `;
}

function formatMetricName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// ---- Info Modal ----
async function showInfo() {
    const modal = document.getElementById('infoModal');
    const content = document.getElementById('infoContent');
    modal.classList.remove('hidden');

    content.innerHTML = '<div class="spinner"></div><p>Loading...</p>';

    try {
        const response = await fetch('/api/info');
        const data = await response.json();

        content.innerHTML = `
            <div style="line-height: 2;">
                <p><strong>System:</strong> ${data.name} v${data.version}</p>
                <p><strong>Project:</strong> ${data.project}</p>
                <p><strong>University:</strong> ${data.university}</p>
                <p><strong>Supervisor:</strong> ${data.supervisor}</p>
                <hr style="margin: 12px 0; border-color: var(--border);">
                <p><strong>Team Members:</strong></p>
                <ul style="padding-left: 20px;">
                    ${data.team.map(m => `<li>${m}</li>`).join('')}
                </ul>
                <hr style="margin: 12px 0; border-color: var(--border);">
                <p><strong>Emotion Labels:</strong> ${data.emotion_labels.join(', ')}</p>
                <p><strong>Models Loaded:</strong> ${data.models_loaded ? '✅ Yes' : '❌ No'}</p>
            </div>
        `;
    } catch (err) {
        content.innerHTML = '<p style="color: var(--sarcastic);">⚠️ Failed to load system info.</p>';
    }
}

// ---- Modal Close ----
function closeModal(modalId) {
    document.getElementById(modalId).classList.add('hidden');
}

// Close modals on outside click
document.querySelectorAll('.modal').forEach(modal => {
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.add('hidden');
        }
    });
});

// Close modals on Escape
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        document.querySelectorAll('.modal').forEach(m => m.classList.add('hidden'));
    }
});
