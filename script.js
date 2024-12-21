const API_BASE = 'https://nicee.up.railway.app';
const style = document.createElement('style');
style.textContent = `
  .qt-menu{position:fixed;top:20px;left:20px;z-index:999999;background:linear-gradient(145deg,rgba(15,15,15,.97),rgba(25,25,25,.97));border:1px solid rgba(255,215,0,.15);border-radius:16px;padding:16px;width:340px;color:#fff;font-family:system-ui;box-shadow:0 8px 32px rgba(0,0,0,.4),0 0 20px rgba(255,215,0,.1);backdrop-filter:blur(12px);transition:all .3s ease;animation:qtFadeIn .3s}
  .qt-menu *{box-sizing:border-box}
  @keyframes qtFadeIn{from{opacity:0;transform:translateY(-20px)}to{opacity:1;transform:translateY(0)}}
  @keyframes qtPulse{0%{box-shadow:0 0 0 0 rgba(255,215,0,.4)}70%{box-shadow:0 0 0 10px rgba(255,215,0,0)}100%{box-shadow:0 0 0 0 rgba(255,215,0,0)}}
  .qt-menu.qt-expanded{width:680px;height:580px}
  .qt-header{display:flex;justify-content:space-between;align-items:center;margin:-16px -16px 16px;padding:16px;border-bottom:1px solid rgba(255,215,0,.15);background:linear-gradient(to right,rgba(255,215,0,.05),transparent);border-radius:16px 16px 0 0;cursor:move;position:relative}
  .qt-title{font-weight:500;color:rgba(255,215,0,.9);font-size:15px;display:flex;align-items:center;gap:8px}
  .qt-stats{position:absolute;right:50px;top:50%;transform:translateY(-50%);text-align:right}
  .qt-balance{color:rgba(100,255,100,.9);font-weight:500;font-size:14px;text-shadow:0 0 10px rgba(100,255,100,.3)}
  .qt-usage{font-size:11px;color:rgba(255,215,0,.6);margin-top:2px}
  .qt-tabs{display:flex;gap:8px;margin-bottom:16px;padding:0 2px}
  .qt-tab{padding:8px 16px;border-radius:8px;cursor:pointer;color:rgba(255,215,0,.6);transition:all .2s;font-size:13px;position:relative}
  .qt-tab.active{background:rgba(255,215,0,.15);color:rgba(255,215,0,.95);box-shadow:0 0 20px rgba(255,215,0,.1)}
  .qt-tab:hover:not(.active){background:rgba(255,215,0,.05)}
  .qt-btn{background:rgba(255,215,0,.1);border:1px solid rgba(255,215,0,.2);color:rgba(255,215,0,.9);padding:8px 16px;border-radius:8px;cursor:pointer;transition:all .2s;font-size:13px;display:inline-flex;align-items:center;gap:6px}
  .qt-btn:hover{background:rgba(255,215,0,.2);transform:translateY(-1px);box-shadow:0 4px 12px rgba(0,0,0,.2)}
  .qt-btn.qt-active{animation:qtPulse 2s infinite;background:rgba(255,215,0,.2)}
  .qt-result{margin-top:16px;padding:16px;background:rgba(255,215,0,.03);border-radius:12px;border:1px solid rgba(255,215,0,.1);display:none}
  .qt-confidence{height:4px;background:rgba(255,215,0,.1);border-radius:2px;margin-top:12px;overflow:hidden}
  .qt-confidence-fill{height:100%;background:linear-gradient(to right,#ffd700,#ffa500);width:0%;transition:width .5s ease}
  .qt-highlight{outline:2px solid rgba(255,165,0,.8)!important;outline-offset:2px;box-shadow:0 0 20px rgba(255,165,0,.2)!important}
  .qt-selected{outline:2px solid rgba(100,255,100,.8)!important;outline-offset:2px;box-shadow:0 0 20px rgba(100,255,100,.2)!important}
  .qt-controls{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}
  .qt-settings{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:4px}
  .qt-switch{display:flex;align-items:center;gap:8px;color:rgba(255,255,255,.9);font-size:13px}
  .qt-switch input{display:none}
  .qt-switch-slider{width:36px;height:20px;background:rgba(255,215,0,.1);border-radius:10px;position:relative;cursor:pointer}
  .qt-switch-slider:after{content:'';width:16px;height:16px;background:rgba(255,215,0,.8);border-radius:50%;position:absolute;top:2px;left:2px;transition:.3s}
  .qt-switch input:checked + .qt-switch-slider{background:rgba(255,215,0,.2)}
  .qt-switch input:checked + .qt-switch-slider:after{left:18px}
  .qt-chat{display:none;flex-direction:column;height:calc(100% - 80px);margin-top:12px}
  .qt-messages{flex:1;overflow-y:auto;padding:8px 4px;display:flex;flex-direction:column;gap:8px}
  .qt-message{padding:10px;border-radius:12px;max-width:85%;animation:qtFadeIn .2s}
  .qt-user{background:rgba(255,215,0,.1);align-self:flex-end;border:1px solid rgba(255,215,0,.15)}
  .qt-bot{background:rgba(255,255,255,.05);align-self:flex-start;border:1px solid rgba(255,255,255,.1)}
  .qt-input-wrap{margin-top:12px;display:flex;gap:8px}
  .qt-chat-input{flex:1;background:rgba(255,255,255,.05);border:1px solid rgba(255,215,0,.15);padding:10px;border-radius:8px;color:white;font-size:13px}
  .qt-shortcuts{position:absolute;bottom:16px;left:16px;font-size:11px;color:rgba(255,215,0,.4)}
  .qt-timer{position:absolute;top:16px;right:16px;font-size:12px;color:rgba(255,215,0,.8)}
  .qt-toast{position:fixed;bottom:20px;right:20px;padding:12px 20px;background:rgba(15,15,15,.95);border:1px solid rgba(255,215,0,.2);border-radius:8px;color:white;font-size:13px;animation:qtFadeIn .3s;z-index:999999}
`;
document.head.appendChild(style);

class ProSolver {
  constructor() {
    this.initState();
    this.createMenu();
    this.setupEventListeners();
    this.loadSettings();
  }

  initState() {
    this.state = {
      auth: { token: null, user: null, credits: 0 },
      solving: { auto: false, selecting: false, timer: null },
      settings: {
        autoDetect: false, autoNext: true, multiQuestion: false,
        showExplanation: true, delay: 3, model: 'gpt4o',
        smartHighlight: true, autoExpand: true, soundEffects: true,
        darkMode: true, keyboardShortcuts: true
      },
      chat: { active: false, history: [], expanded: false },
      stats: { solved: 0, accuracy: 0, streak: 0 }
    };
  }

  createMenu() {
    const menu = document.createElement('div');
    menu.className = 'qt-menu';
    menu.innerHTML = `
      <div class="qt-header">
        <div class="qt-title">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="rgba(255,215,0,.9)">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
          </svg>
          Pro Solver
        </div>
        <div class="qt-stats">
          <div class="qt-balance">$0.00</div>
          <div class="qt-usage">Solved: 0</div>
        </div>
        <button class="qt-btn" style="padding:4px 8px">×</button>
      </div>
      <div class="qt-tabs">
        <div class="qt-tab active" data-tab="solve">Solve</div>
        <div class="qt-tab" data-tab="config">Config</div>
        <div class="qt-tab" data-tab="custom">Chat</div>
      </div>
      <div class="qt-content">
        <div class="qt-tab-content active" data-tab="solve">
          <div class="qt-controls">
            <button class="qt-btn" id="qt-select">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <path d="M4 4h16v16H4z"/>
              </svg>
              Select Question
            </button>
            <button class="qt-btn" id="qt-auto">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <path d="M8 5v14l11-7z"/>
              </svg>
              Start Auto
            </button>
          </div>
          <div class="qt-result">
            <div class="qt-answer"></div>
            <div class="qt-explain"></div>
            <div class="qt-confidence"><div class="qt-confidence-fill"></div></div>
          </div>
          <div class="qt-timer" style="display:none">Next: 3s</div>
        </div>
        <div class="qt-tab-content" data-tab="config">
          <div class="qt-settings">
            ${this.createSettingsSwitches()}
          </div>
        </div>
        <div class="qt-tab-content" data-tab="custom">
          <div class="qt-chat"></div>
        </div>
      </div>
      <div class="qt-shortcuts">Alt+A: Auto • Alt+S: Select • Alt+C: Chat</div>
    `;
    document.body.appendChild(menu);
    this.menu = menu;
  }

  createSettingsSwitches() {
    return Object.entries(this.state.settings)
      .map(([key, value]) => `
        <label class="qt-switch">
          <input type="checkbox" id="qt-${key}" ${value ? 'checked' : ''}>
          <span class="qt-switch-slider"></span>
          ${key.replace(/([A-Z])/g, ' $1').toLowerCase()}
        </label>
      `).join('');
  }

  async solve(question) {
    try {
      const response = await fetch(`${API_BASE}/api/solve`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.state.auth.token}`
        },
        body: JSON.stringify({
          text: question.textContent,
          context: this.getQuestionContext(question),
          model: this.state.settings.model
        })
      });
      
      if (!response.ok) throw new Error('Failed to solve');
      const data = await response.json();
      
      this.updateStats(data);
      this.showResult(data);
      
      if (this.state.settings.autoNext) {
        setTimeout(this.clickNext.bind(this), 1000);
      }
    } catch (error) {
      this.showToast(`Error: ${error.message}`);
    }
  }

  getQuestionContext(el) {
    const context = { text: el.textContent, images: [], nearbyText: [] };
    
    // Get nearby images
    el.closest('div')?.querySelectorAll('img').forEach(img => {
      context.images.push({ src: img.src, alt: img.alt });
    });
    
    // Get surrounding text context
    el.parentElement?.childNodes.forEach(node => {
      if (node.nodeType === 3 && node.textContent.trim()) {
        context.nearbyText.push(node.textContent.trim());
      }
    });
    
    return context;
  }

  updateStats({ cost, confidence }) {
    this.state.stats.solved++;
    this.state.auth.credits -= cost;
    
    const balanceEl = this.menu.querySelector('.qt-balance');
    balanceEl.textContent = `$${this.state.auth.credits.toFixed(2)}`;
    balanceEl.style.color = this.state.auth.credits < 1 ? '#ff6b6b' : '#90ee90';
    
    this.menu.querySelector('.qt-usage').textContent = 
      `Solved: ${this.state.stats.solved} • Streak: ${this.state.stats.streak}`;
    
    this.menu.querySelector('.qt-confidence-fill').style.width = `${confidence}%`;
  }

  showResult({ answer, explanation, confidence }) {
    const result = this.menu.querySelector('.qt-result');
    result.style.display = 'block';
    result.querySelector('.qt-answer').innerHTML = `<strong>Answer:</strong> ${answer}`;
    
    if (this.state.settings.showExplanation) {
      result.querySelector('.qt-explain').innerHTML = 
        `<div style="margin-top:8px"><strong>Explanation:</strong> ${explanation}</div>`;
    }
    
    if (this.state.settings.soundEffects) {
      const successSound = new Audio('data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4LjIuMTAwAAAAAAAAAAAAAAAA//tUZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAASW5mbwAAAA8AAAAGAAADQAAYGBgYJCQkJCQwMDAwMDw8PDw8SEhISEhUVFRUVGBgYGBgbGxsbGyIiIiIiJSUlJSUoKCgoKCsrKysrLi4uLi4xMTExMTQ0NDQ0P8AAABQTEFNRTMuMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV//tUZAAAAaYA0e0AAAgAAA/woAABGMmzV7mngCEAACXDAAAAVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVf/7UGQAAAZcZ1O0zngIQAAN8KAAARMR1W7TOeAhAAA/woAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVX/+1BkAAAGPGdVtM54CEAAD/CgAAETUd1m0zngIQAAP8KAAAQAAFXBQUUAALODg4KCgoCgAAAAoKDg4KCgKBQAABQUFBwcHBQAAoKDg4ODg4AABQUHBQUHBQVVVVVVVVVVVVX/+1BkAAAGOHVdtM54CEAAD/CgAAEScd1+0zngIQAAP8KAAAQVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV//tQZAAABlB5X7TOeAhAAA/woAABE1HdftM54CEAAD/CgAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVf/7UGQAAAYUd1+0zngIQAAP8KAAARM53W7TOeAhAAA/woAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVX/+1BkAAAGwHdfvM54CEAAD/CgAAEUYWN+1HOAAgAAA/w4AABVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV//tQZAAABkB3X7TOeAhAAA/woAABE7ndfuj4ACEAAD/DgAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVf/7UGQAAAXwRN21LOAAgAAA/w4AABGFHdfvM54CEAAD/CgAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVX/+1BkAAAF/3dfvM54CEAAD/CgAAESieVW6Z7wEAAAD/DgAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV').play();
    }

    result.classList.add('qt-highlight');
    setTimeout(() => result.classList.remove('qt-highlight'), 500);
  }

  findQuestionElements() {
    const questionSelectors = [
      '#questiontext', '.question-text', '[role="heading"]',
      '.prose', '[data-test="question"]', '.boxed'
    ];
    
    const answerSelectors = [
      '.btn-answer', '[role="option"]', 'input[type="radio"]',
      '.answer-choice', 'li.btn', '[data-answerid]'
    ];

    const question = questionSelectors.reduce((found, selector) => 
      found || document.querySelector(selector), null);

    const answers = answerSelectors.reduce((found, selector) => 
      found.length ? found : Array.from(document.querySelectorAll(selector)), []);

    return { question, answers };
  }

  async autoSolve() {
    if (!this.state.solving.auto) return;
    
    const { question, answers } = this.findQuestionElements();
    if (!question || !answers.length) {
      this.showToast('No question or answers found');
      return;
    }

    question.classList.add('qt-highlight');
    await this.solve(question);
    question.classList.remove('qt-highlight');

    const timer = setInterval(() => {
      const remaining = Math.ceil(this.state.settings.delay - 
        ((Date.now() - startTime) / 1000));
      this.menu.querySelector('.qt-timer').textContent = `Next: ${remaining}s`;
      if (remaining <= 0) {
        clearInterval(timer);
        this.autoSolve();
      }
    }, 1000);
  }

  clickNext() {
    const nextSelectors = [
      '#lnkNext', '.btn-success', '[aria-label*="next"]',
      'button:contains("Continue")', '[role="button"]:contains("Next")'
    ];

    for (const selector of nextSelectors) {
      const button = document.querySelector(selector);
      if (button) {
        button.click();
        return true;
      }
    }
    return false;
  }

  showToast(message, duration = 3000) {
    const toast = document.createElement('div');
    toast.className = 'qt-toast';
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), duration);
  }

  setupEventListeners() {
    // Tab switching
    this.menu.querySelectorAll('.qt-tab').forEach(tab => {
      tab.onclick = () => {
        this.menu.querySelectorAll('.qt-tab, .qt-tab-content').forEach(el => 
          el.classList.remove('active'));
        tab.classList.add('active');
        this.menu.querySelector(`.qt-tab-content[data-tab="${tab.dataset.tab}"]`)
          .classList.add('active');
        if (tab.dataset.tab === 'custom') {
          this.menu.classList.add('qt-expanded');
        } else {
          this.menu.classList.remove('qt-expanded');
        }
      };
    });

    // Auto solve button
    const autoBtn = this.menu.querySelector('#qt-auto');
    autoBtn.onclick = () => {
      this.state.solving.auto = !this.state.solving.auto;
      autoBtn.classList.toggle('qt-active');
      autoBtn.innerHTML = this.state.solving.auto ? 
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M6 4h4v16H6zm8 0h4v16h-4z"/></svg> Stop Auto' :
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg> Start Auto';
      
      if (this.state.solving.auto) {
        this.autoSolve();
        this.menu.querySelector('.qt-timer').style.display = 'block';
      } else {
        this.menu.querySelector('.qt-timer').style.display = 'none';
      }
    };

    // Settings
    this.menu.querySelectorAll('.qt-switch input').forEach(input => {
      input.onchange = () => {
        const key = input.id.replace('qt-', '');
        this.state.settings[key] = input.checked;
        localStorage.setItem('qt-settings', JSON.stringify(this.state.settings));
      };
    });

    // Make menu draggable
    const header = this.menu.querySelector('.qt-header');
    header.onmousedown = e => {
      if (e.target.tagName === 'BUTTON') return;
      
      const startPos = { x: e.clientX - this.menu.offsetLeft,
                        y: e.clientY - this.menu.offsetTop };
      
      const onMouseMove = e => {
        this.menu.style.left = `${e.clientX - startPos.x}px`;
        this.menu.style.top = `${e.clientY - startPos.y}px`;
      };
      
      const onMouseUp = () => {
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
      };
      
      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    };

    // Keyboard shortcuts
    document.addEventListener('keydown', e => {
      if (!this.state.settings.keyboardShortcuts) return;
      
      if (e.altKey) {
        switch(e.key.toLowerCase()) {
          case 'a':
            autoBtn.click();
            break;
          case 's':
            this.menu.querySelector('#qt-select').click();
            break;
          case 'c':
            this.menu.querySelector('[data-tab="custom"]').click();
            break;
        }
      }
    });
  }

  loadSettings() {
    const saved = localStorage.getItem('qt-settings');
    if (saved) {
      this.state.settings = { ...this.state.settings, ...JSON.parse(saved) };
      Object.entries(this.state.settings).forEach(([key, value]) => {
        const input = this.menu.querySelector(`#qt-${key}`);
        if (input) input.checked = value;
      });
    }
  }
}

// Initialize
new ProSolver();
