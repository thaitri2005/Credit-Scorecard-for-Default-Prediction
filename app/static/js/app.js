// Credit Risk Scorecard UI

const NUMERIC_FIELDS = [
    'annual_inc', 'credit_history_length', 'loan_amount', 'int_rate',
    'total_rev_hi_lim', 'tot_cur_bal', 'revol_util',
];
const REQUIRED_FIELDS = [
    'annual_inc', 'credit_history_length', 'verification_status',
    'loan_amount', 'int_rate', 'purpose',
];
// Fields the API requires to be strictly greater than zero.
const STRICTLY_POSITIVE = ['annual_inc', 'loan_amount'];

const PRESETS = {
    strong: {
        annual_inc: 140000, credit_history_length: 22, verification_status: 'Not Verified',
        loan_amount: 12000, int_rate: 6.9, purpose: 'credit_card',
        total_rev_hi_lim: 60000, tot_cur_bal: 310000, revol_util: 18,
    },
    typical: {
        annual_inc: 75000, credit_history_length: 5.5, verification_status: 'Verified',
        loan_amount: 15000, int_rate: 12.5, purpose: 'debt_consolidation',
        total_rev_hi_lim: 30000, tot_cur_bal: 120000, revol_util: 40,
    },
    stretched: {
        annual_inc: 38000, credit_history_length: 4, verification_status: 'Verified',
        loan_amount: 18000, int_rate: 21.5, purpose: 'small_business',
        total_rev_hi_lim: 9000, tot_cur_bal: 21000, revol_util: 92,
    },
};

// Breakdown rows follow the order of the form.
const FEATURE_ORDER = [
    'annual_inc', 'credit_history_length', 'verification_status',
    'loan_burden', 'int_rate', 'purpose',
    'total_rev_hi_lim', 'tot_cur_bal', 'revol_util',
];

// Scale shown when the model info endpoint cannot be reached.
const FALLBACK_BANDS = [
    { min_score: 620, rating: 'AAA' }, { min_score: 605, rating: 'AA' },
    { min_score: 590, rating: 'A' }, { min_score: 575, rating: 'BBB' },
    { min_score: 560, rating: 'BB' }, { min_score: 545, rating: 'B' },
    { min_score: 530, rating: 'CCC' }, { min_score: 515, rating: 'CC' },
];

const money = new Intl.NumberFormat('en-US', {
    style: 'currency', currency: 'USD', maximumFractionDigits: 0,
});
const percent = (value, digits = 1) => `${Number(value).toFixed(digits)}%`;
const signed = (value) => `${value > 0 ? '+' : value < 0 ? '−' : ''}${Math.abs(value).toFixed(1)}`;

class CreditRiskUI {
    constructor() {
        this.form = document.getElementById('loanForm');
        this.results = document.getElementById('results');
        this.resultBody = document.getElementById('resultBody');
        this.resultEmpty = document.getElementById('resultEmpty');
        this.resultError = document.getElementById('resultError');
        this.mobileBar = document.getElementById('mobileBar');
        this.chips = Array.from(document.querySelectorAll('.chip'));
        this.bands = [];
        this.pending = null;
        this.timer = null;

        this.form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.score({ showErrors: true, focusResult: true });
        });
        document.getElementById('clearBtn').addEventListener('click', () => this.clear());
        this.chips.forEach((chip) => {
            chip.addEventListener('click', () => this.applyPreset(chip.dataset.preset));
        });
        this.fields().forEach((field) => {
            field.addEventListener('input', () => {
                field.dataset.edited = 'true';
                this.selectChip(null);
                this.setFieldError(field, '');
                clearTimeout(this.timer);
                this.timer = setTimeout(() => this.score({ showErrors: false }), 300);
            });
            field.addEventListener('blur', () => {
                if (field.value.trim() || field.dataset.edited) {
                    this.setFieldError(field, this.fieldError(field));
                }
            });
        });

        this.themeToggle = document.getElementById('themeToggle');
        this.themeToggle.addEventListener('click', () => {
            const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
            this.setTheme(next);
            try {
                localStorage.setItem('theme', next);
            } catch (error) {
                // The choice simply lasts for this visit.
            }
        });
        let savedTheme = null;
        try {
            savedTheme = localStorage.getItem('theme');
        } catch (error) {
            savedTheme = null;
        }
        this.setTheme(savedTheme || document.documentElement.dataset.theme || 'light');

        // The bottom summary bar is only useful while the assessment is off screen.
        if ('IntersectionObserver' in window) {
            new IntersectionObserver(([entry]) => {
                this.mobileBar.classList.toggle('offscreen', entry.isIntersecting);
            }, { threshold: 0.2 }).observe(this.results);
        }

        this.buildScale(FALLBACK_BANDS, 'C');
        this.applyPreset('typical');
        this.refreshScale();
    }

    setTheme(theme) {
        document.documentElement.dataset.theme = theme;
        this.themeToggle.textContent = theme === 'dark' ? 'Light mode' : 'Dark mode';
    }

    fields() {
        return Array.from(this.form.querySelectorAll('input, select'));
    }

    labelFor(name) {
        const label = this.form.querySelector(`label[for="${name}"]`);
        return label ? label.textContent.trim() : name;
    }

    // ----- Scale -----

    // Replace the built-in scale with the model's own bands once they arrive.
    async refreshScale() {
        try {
            const response = await fetch('/api/v1/model/info');
            if (!response.ok) {
                return;
            }
            const info = await response.json();
            if (info.rating_bands && info.rating_bands.length) {
                this.buildScale(info.rating_bands, info.lowest_rating || 'C');
                if (this.last) {
                    this.renderScale(this.last.score, this.last.rating);
                }
            }
        } catch (error) {
            // Keep the built-in scale.
        }
    }

    buildScale(bands, lowest) {
        // Worst grade first; each band spans [from, to) in score points.
        const ascending = [...bands].sort((a, b) => a.min_score - b.min_score);
        const step = ascending.length > 1 ? ascending[1].min_score - ascending[0].min_score : 15;
        this.bands = [{ rating: lowest, from: ascending[0].min_score - step }];
        ascending.forEach((band) => this.bands.push({ rating: band.rating, from: band.min_score }));
        this.bands.forEach((band, i) => {
            const next = this.bands[i + 1];
            band.to = next ? next.from : band.from + 2 * step;
        });
        this.scaleMin = this.bands[0].from;
        this.scaleMax = this.bands[this.bands.length - 1].to;

        const bandsEl = document.getElementById('scaleBands');
        const axisEl = document.getElementById('scaleAxis');
        bandsEl.replaceChildren();
        axisEl.replaceChildren();
        this.bands.forEach((band) => {
            const segment = document.createElement('div');
            segment.className = `scale-band grade-${band.rating}`;
            segment.style.flex = `${band.to - band.from} 1 0`;
            segment.dataset.rating = band.rating;
            bandsEl.appendChild(segment);

            const label = document.createElement('span');
            label.textContent = band.rating;
            label.style.flex = `${band.to - band.from} 1 0`;
            label.dataset.rating = band.rating;
            axisEl.appendChild(label);
        });
    }

    renderScale(score, rating) {
        const clamped = Math.min(Math.max(score, this.scaleMin), this.scaleMax);
        const position = ((clamped - this.scaleMin) / (this.scaleMax - this.scaleMin)) * 100;
        document.getElementById('scaleMarker').style.left = `${position}%`;
        document.querySelectorAll('#scaleBands > *, #scaleAxis > *').forEach((el) => {
            el.classList.toggle('current', el.dataset.rating === rating);
        });
        const index = this.bands.findIndex((b) => b.rating === rating);
        const band = this.bands[index];
        let range = '';
        if (band && index === 0) {
            range = `scores below ${band.to}`;
        } else if (band && index === this.bands.length - 1) {
            range = `scores of ${band.from} and above`;
        } else if (band) {
            range = `scores from ${band.from} to ${band.to - 1}`;
        }
        document.getElementById('scaleCaption').textContent = range
            ? `Grade ${rating} covers ${range}.`
            : '';
        document.getElementById('scale').setAttribute(
            'aria-label', `Score ${Math.floor(score)} on a scale of grades from ${this.bands[0].rating} to ${this.bands[this.bands.length - 1].rating}`
        );
    }

    // ----- Validation -----

    fieldError(field) {
        const raw = field.value.trim();
        if (field.type === 'number' && field.validity.badInput) {
            return 'Enter a number';
        }
        if (!raw) {
            return REQUIRED_FIELDS.includes(field.name) ? 'Required' : '';
        }
        if (field.type !== 'number') {
            return '';
        }
        const value = Number(raw);
        if (!Number.isFinite(value)) {
            return 'Enter a number';
        }
        if (STRICTLY_POSITIVE.includes(field.name) && value <= 0) {
            return 'Must be greater than 0';
        }
        if (field.min !== '' && value < Number(field.min)) {
            return `Must be at least ${field.min}`;
        }
        if (field.max !== '' && value > Number(field.max)) {
            return `Must be at most ${Number(field.max).toLocaleString('en-US')}`;
        }
        return '';
    }

    setFieldError(field, message) {
        const hint = document.getElementById(`${field.id}_hint`);
        field.closest('.control').classList.toggle('invalid', Boolean(message));
        hint.classList.toggle('error', Boolean(message));
        hint.dataset.error = message;
        if (message) {
            field.setAttribute('aria-invalid', 'true');
        } else {
            field.removeAttribute('aria-invalid');
        }
    }

    // ----- Scoring -----

    payload() {
        const data = {};
        this.fields().forEach((field) => {
            const raw = field.value.trim();
            if (raw) {
                data[field.name] = NUMERIC_FIELDS.includes(field.name) ? Number(raw) : raw;
            }
        });
        return data;
    }

    async score({ showErrors, focusResult = false }) {
        clearTimeout(this.timer);
        const errors = this.fields().map((field) => [field, this.fieldError(field)]);
        const invalid = errors.filter(([, message]) => message);
        if (showErrors) {
            errors.forEach(([field, message]) => this.setFieldError(field, message));
        }
        if (invalid.length) {
            this.showEmpty();
            if (showErrors) {
                invalid[0][0].focus();
            }
            return;
        }

        if (this.pending) {
            this.pending.abort();
        }
        const controller = new AbortController();
        this.pending = controller;
        const payload = this.payload();
        this.results.setAttribute('aria-busy', 'true');

        try {
            const response = await fetch('/api/v1/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: controller.signal,
            });
            let body = null;
            try {
                body = await response.json();
            } catch (parseError) {
                body = null;
            }
            if (!response.ok || !body) {
                this.showError(this.describeError(response.status, body));
                if (focusResult) {
                    this.results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
                return;
            }
            this.render(body, payload);
            if (focusResult) {
                this.results.focus({ preventScroll: true });
                this.results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        } catch (error) {
            if (error.name !== 'AbortError') {
                this.showError('The scoring service could not be reached. Check the connection and resubmit.');
            }
        } finally {
            if (this.pending === controller) {
                this.pending = null;
                this.results.removeAttribute('aria-busy');
            }
        }
    }

    // Reword an API validation message for display beside a field.
    plainMessage(message) {
        const text = String(message).replace(/^Value error, /, '');
        const rules = [
            [/^Input should be less than or equal to (.+)$/, 'Must be at most $1'],
            [/^Input should be greater than or equal to (.+)$/, 'Must be at least $1'],
            [/^Input should be greater than (.+)$/, 'Must be greater than $1'],
            [/^Input should be a valid number.*$/, 'Enter a number'],
            [/^Input should be a finite number$/, 'Enter a number'],
            [/^Field required$/, 'Required'],
        ];
        for (const [pattern, replacement] of rules) {
            if (pattern.test(text)) {
                return text.replace(pattern, replacement);
            }
        }
        return text;
    }

    // Turn an API error into a message, marking the fields it names
    // (a 422 from the API carries a list of {loc, msg}).
    describeError(status, body) {
        const detail = body && body.detail;
        if (Array.isArray(detail)) {
            const messages = detail.map((item) => {
                const name = Array.isArray(item.loc) ? item.loc[item.loc.length - 1] : null;
                const field = name ? document.getElementById(name) : null;
                const message = this.plainMessage(item.msg);
                if (field && this.form.contains(field)) {
                    this.setFieldError(field, message);
                    return `${this.labelFor(name)}: ${message}`;
                }
                return message;
            });
            return messages.join('. ') || 'One or more application details are not valid.';
        }
        if (status >= 200 && status < 300) {
            return 'The scoring service returned an unreadable response. Resubmit the application.';
        }
        return `The assessment could not be completed (error ${status}). Resubmit the application; `
            + 'if the problem persists, contact the system administrator.';
    }

    // ----- Rendering -----

    describeInput(feature, payload) {
        const value = feature === 'loan_burden' ? payload.loan_amount : payload[feature];
        if (value === undefined) {
            return 'Not provided';
        }
        switch (feature) {
            case 'int_rate':
            case 'revol_util':
                return percent(value, 1);
            case 'credit_history_length':
                return `${value} ${value === 1 ? 'year' : 'years'}`;
            case 'loan_burden': {
                const share = (payload.loan_amount / (payload.annual_inc + 1)) * 100;
                return share > 999 ? 'Over 999% of income' : `${percent(share, 0)} of income`;
            }
            case 'purpose':
            case 'verification_status': {
                const option = this.form.querySelector(`#${feature} option[value="${value}"]`);
                return option ? option.textContent : value;
            }
            default:
                return money.format(value);
        }
    }

    render(result, payload) {
        const score = Number(result.credit_score);
        // Scores are shown rounded down so a figure never appears to cross a
        // grade threshold it has not reached.
        const wholeScore = String(Math.floor(score));
        this.last = { score, rating: result.risk_level };
        document.getElementById('creditScore').textContent = wholeScore;
        document.getElementById('defaultProb').textContent = percent(result.default_probability * 100, 1);
        const grade = document.getElementById('riskLevel');
        grade.textContent = result.risk_level;
        grade.className = `grade grade-${result.risk_level}`;
        document.getElementById('riskDescription').textContent = result.risk_description || '';
        this.renderScale(score, result.risk_level);

        const ordered = [...(result.points || [])].sort(
            (a, b) => FEATURE_ORDER.indexOf(a.feature) - FEATURE_ORDER.indexOf(b.feature)
        );
        // Bars share one scale so their lengths compare across rows.
        const largest = Math.max(10, ...ordered.map((item) => Math.abs(item.points)));

        const rows = document.getElementById('breakdownRows');
        rows.replaceChildren();
        const addRow = (label, attribute, points, text) => {
            const row = document.createElement('tr');
            const name = document.createElement('th');
            name.scope = 'row';
            const title = document.createElement('span');
            title.textContent = label;
            const detail = document.createElement('span');
            detail.className = 'attribute';
            detail.textContent = attribute;
            name.append(title, detail);

            const barCell = document.createElement('td');
            barCell.className = 'bar-cell';
            barCell.setAttribute('aria-hidden', 'true');
            if (points !== null) {
                const bar = document.createElement('span');
                bar.className = `bar ${points >= 0 ? 'pos' : 'neg'}`;
                bar.style.width = `${(Math.abs(points) / largest) * 50}%`;
                barCell.appendChild(bar);
            }

            const number = document.createElement('td');
            number.className = 'num';
            if (points !== null && Math.abs(points) > 0.05) {
                number.classList.add(points > 0 ? 'pos' : 'neg');
            }
            number.textContent = text;
            row.append(name, barCell, number);
            rows.appendChild(row);
        };
        addRow('Base score', 'Population average', null, Number(result.base_points).toFixed(1));
        ordered.forEach((item) => {
            const label = item.feature === 'loan_burden' ? 'Loan amount' : this.labelFor(item.feature);
            addRow(label, this.describeInput(item.feature, payload), item.points, signed(item.points));
        });
        document.getElementById('breakdownTotal').textContent = (Math.floor(score * 10) / 10).toFixed(1);

        const missing = (result.missing_fields || []).map((name) => this.labelFor(name));
        document.getElementById('resultMessage').textContent = missing.length
            ? `Not provided; assigned the least favorable attribute: ${missing.join(', ')}.`
            : '';

        this.setSummary(wholeScore, result.risk_level);

        this.resultError.hidden = true;
        this.resultEmpty.hidden = true;
        this.resultBody.hidden = false;
    }

    // The summary bar pinned to the bottom of narrow screens.
    setSummary(score, status) {
        document.getElementById('mobileScore').textContent = score;
        document.getElementById('mobileGrade').textContent = status;
        this.mobileBar.hidden = false;
        document.body.classList.add('has-result');
    }

    showEmpty() {
        if (this.pending) {
            this.pending.abort();
        }
        this.last = null;
        this.resultBody.hidden = true;
        this.resultError.hidden = true;
        this.resultEmpty.hidden = false;
        this.setSummary('', 'Application incomplete');
    }

    showError(message) {
        this.last = null;
        this.resultBody.hidden = true;
        this.resultEmpty.hidden = true;
        this.resultError.textContent = message;
        this.resultError.hidden = false;
        this.setSummary('', 'Assessment unavailable');
    }

    // ----- Presets -----

    selectChip(name) {
        this.chips.forEach((chip) => {
            chip.setAttribute('aria-pressed', String(chip.dataset.preset === name));
        });
    }

    applyPreset(name) {
        Object.entries(PRESETS[name]).forEach(([key, value]) => {
            const field = document.getElementById(key);
            field.value = value;
            delete field.dataset.edited;
            this.setFieldError(field, '');
        });
        this.selectChip(name);
        this.score({ showErrors: false });
    }

    clear() {
        this.form.reset();
        this.fields().forEach((field) => {
            delete field.dataset.edited;
            this.setFieldError(field, '');
        });
        this.selectChip(null);
        this.showEmpty();
        document.getElementById('annual_inc').focus();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new CreditRiskUI();
});
