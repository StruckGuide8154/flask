javascript:(()=>{
  if(document.querySelector('#qt-app')){document.querySelector('#qt-app').remove();return;}
  const API='https://nicee.up.railway.app',style=document.createElement('style');
  style.textContent=`
    #qt-app{position:fixed;top:20px;left:20px;z-index:999999;background:linear-gradient(145deg,#191919fa,#0f0f0ffa);border:1px solid #ffd70026;border-radius:12px;padding:15px;width:320px;color:#fff;font-family:system-ui;box-shadow:0 8px 32px #0006;backdrop-filter:blur(12px)}
    .qt-head{display:flex;justify-content:space-between;align-items:center;margin:-15px -15px 10px;padding:12px;border-bottom:1px solid #ffd70026;cursor:move}
    .qt-btn{background:#ffd70019;border:1px solid #ffd70033;color:#ffd700cc;padding:7px 12px;border-radius:6px;cursor:pointer;font-size:13px}
    .qt-btn:hover{background:#ffd70033;transform:translateY(-1px)}
    .qt-btn:disabled{opacity:.5;cursor:not-allowed;transform:none}
    .qt-sel{outline:2px solid #ffd700!important;outline-offset:2px!important}
    .qt-title{color:#ffd700e6;font-weight:500;font-size:14px}
    .qt-result{margin-top:12px;padding:10px;background:#ffd70009;border-radius:6px;font-size:13px}
    .qt-model{width:100%;background:#ffffff1a;border:1px solid #ffd70033;padding:7px;border-radius:6px;color:#fff;margin:8px 0}
    .qt-conf{height:4px;background:#ffd70019;border-radius:2px;margin-top:8px}
    .qt-conf-fill{height:100%;background:#ffd700cc;width:0%;transition:width .3s}
    .qt-login{background:#ffffff0d;padding:12px;border-radius:6px;margin:10px 0}
    .qt-input{width:100%;background:#ffffff1a;border:1px solid #ffd70033;padding:7px;border-radius:6px;color:#fff;margin:4px 0}
    .qt-error{color:#ff6b6b;font-size:12px;margin-top:8px}
    .qt-credits{color:#4CAF50;font-size:12px;margin-left:12px}
    .qt-files{display:flex;flex-wrap:wrap;gap:4px;margin:8px 0}
    .qt-file{background:#ffffff0d;padding:4px 8px;border-radius:4px;font-size:12px;display:flex;align-items:center}
    .qt-file button{margin-left:6px;opacity:.7;cursor:pointer}
    .qt-file button:hover{opacity:1}`;
  document.head.appendChild(style);

  class Solver {
    constructor() {
      this.init();
      this.token = localStorage.getItem('qt-token');
      this.models = {
        'gpt4o-mini': 'GPT-4 Mini (Fast)',
        'gpt4o': 'GPT-4 (Smart)',
        'claude-haiku': 'Claude Haiku (Fast)',
        'claude-sonnet': 'Claude Sonnet (Smart)',
        'claude-opus': 'Claude Opus (Best)',
        'gemini-flash': 'Gemini Flash (Quick)',
        'gemini-pro': 'Gemini Pro (Balanced)', 
        'gemini-pro-2': 'Gemini Pro 2 (Advanced)'
      };
      this.files = [];
      this.setupUI();
      this.checkAuth();
    }

    init() {
      this.app = document.createElement('div');
      this.app.id = 'qt-app';
      document.body.appendChild(this.app);
    }

    setupUI() {
      this.app.innerHTML = this.token ? this.mainUI() : this.loginUI();
      if(this.token) this.setupMainEvents();
      else this.setupLoginEvents();
    }

    loginUI() {
      return `
        <div class="qt-head">
          <div class="qt-title">Login Required</div>
          <button class="qt-btn" style="padding:4px 8px" onclick="this.parentElement.parentElement.remove()">×</button>
        </div>
        <div class="qt-login">
          <input type="text" class="qt-input" placeholder="Username" id="qt-user">
          <input type="password" class="qt-input" placeholder="Password" id="qt-pass">
          <button class="qt-btn" style="width:100%;margin-top:8px" id="qt-login">Login</button>
          <div class="qt-error" id="qt-error" style="display:none"></div>
        </div>`;
    }

    mainUI() {
      return `
        <div class="qt-head">
          <div style="display:flex;align-items:center">
            <div class="qt-title">Question Solver</div>
            <div class="qt-credits" id="qt-credits"></div>
          </div>
          <button class="qt-btn" style="padding:4px 8px" onclick="this.parentElement.parentElement.remove()">×</button>
        </div>
        <select class="qt-model" id="qt-model">
          ${Object.entries(this.models).map(([k,v])=>`<option value="${k}">${v}</option>`).join('')}
        </select>
        <div class="qt-files" id="qt-files"></div>
        <button class="qt-btn" style="width:100%" id="qt-select">Select Question</button>
        <button class="qt-btn" style="width:100%;margin-top:8px" id="qt-upload">
          Upload File
        </button>
        <div class="qt-result" id="qt-result" style="display:none"></div>
        <div class="qt-conf"><div class="qt-conf-fill" id="qt-conf"></div></div>`;
    }

    setupLoginEvents() {
      const login = document.getElementById('qt-login'),
            user = document.getElementById('qt-user'),
            pass = document.getElementById('qt-pass'),
            error = document.getElementById('qt-error');

      login.onclick = async () => {
        try {
          const res = await fetch(`${API}/api/aii/login`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
              username: user.value,
              password: pass.value
            })
          });
          
          if(!res.ok) throw new Error('Invalid credentials');
          
          const data = await res.json();
          this.token = data.token;
          localStorage.setItem('qt-token', data.token);
          this.setupUI();
          this.checkAuth();
        } catch(e) {
          error.textContent = e.message;
          error.style.display = 'block';
        }
      };
    }

    setupMainEvents() {
      let selecting = false, selected = null;
      const select = document.getElementById('qt-select'),
            result = document.getElementById('qt-result'),
            model = document.getElementById('qt-model'),
            conf = document.getElementById('qt-conf'),
            upload = document.getElementById('qt-upload'),
            filesDiv = document.getElementById('qt-files');

      // Selection logic
      select.onclick = () => {
        selecting = !selecting;
        select.textContent = selecting ? 'Cancel' : 'Select Question';
        if(!selecting) {
          document.querySelectorAll('.qt-highlight').forEach(el => {
            el.classList.remove('qt-highlight');
            el.style.cursor = '';
          });
        }
      };

      // File upload
      upload.onclick = () => {
        const input = document.createElement('input');
        input.type = 'file';
        input.multiple = true;
        input.accept = 'image/*,.pdf,.txt';
        input.onchange = e => {
          Array.from(e.target.files).forEach(file => {
            this.files.push(file);
            const div = document.createElement('div');
            div.className = 'qt-file';
            div.innerHTML = `${file.name}<button onclick="this.parentElement.remove()">×</button>`;
            filesDiv.appendChild(div);
          });
        };
        input.click();
      };

      // Hover effects
      document.addEventListener('mouseover', e => {
        if(!selecting || e.target === this.app || this.app.contains(e.target)) return;
        e.target.classList.add('qt-highlight');
        e.target.style.cursor = 'pointer';
      });

      document.addEventListener('mouseout', e => {
        if(!selecting) return;
        e.target.classList.remove('qt-highlight');
        e.target.style.cursor = '';
      });

      // Question selection
      document.addEventListener('click', async e => {
        if(!selecting || e.target === this.app || this.app.contains(e.target)) return;
        e.preventDefault();
        selecting = false;
        select.textContent = 'Select Question';
        
        if(selected) selected.classList.remove('qt-sel');
        selected = e.target;
        selected.classList.add('qt-sel');

        result.style.display = 'block';
        result.innerHTML = '<div style="text-align:center">Analyzing...</div>';

        try {
          const formData = new FormData();
          formData.append('text', selected.innerText.trim());
          formData.append('model', model.value);
          this.files.forEach(f => formData.append('file', f));

          const res = await fetch(`${API}/api/aii/solve`, {
            method: 'POST',
            headers: {'Authorization': `Bearer ${this.token}`},
            body: formData
          });

          if(res.status === 401) {
            localStorage.removeItem('qt-token');
            this.setupUI();
            return;
          }

          if(res.status === 402) {
            result.innerHTML = '<div class="qt-error">Insufficient credits</div>';
            return;
          }

          if(!res.ok) throw new Error(await res.text());

          const data = await res.json();
          conf.style.width = data.confidence + '%';

          result.innerHTML = `
            <div><b>Answer:</b> ${data.answer}</div>
            <div style="margin-top:8px"><b>Explanation:</b> ${data.explanation}</div>
            <button class="qt-btn" style="width:100%;margin-top:12px" onclick="
              document.querySelectorAll('input[type=text],textarea').forEach(i=>{
                if(i.offsetWidth>0&&i.offsetHeight>0){
                  i.value=\`${data.answer}\`;
                  i.dispatchEvent(new Event('input',{bubbles:true}));
                }
              });
              setTimeout(()=>document.querySelector('button:not(.qt-btn)').click(),500);
            ">Apply Answer</button>`;

          // Update credits
          this.checkAuth();
        } catch(e) {
          result.innerHTML = `<div class="qt-error">Error: ${e.message}</div>`;
        }
      });

      // Make draggable
      let dragging = false;
      this.app.querySelector('.qt-head').onmousedown = e => {
        if(e.target.tagName === 'BUTTON') return;
        dragging = true;
        const rect = this.app.getBoundingClientRect(),
              offX = e.clientX - rect.left,
              offY = e.clientY - rect.top;

        const move = e => {
          if(!dragging) return;
          this.app.style.left = (e.clientX - offX) + 'px';
          this.app.style.top = (e.clientY - offY) + 'px';
        };

        document.onmousemove = move;
        document.onmouseup = () => {
          dragging = false;
          document.onmousemove = null;
        };
      };
    }

    async checkAuth() {
      if(!this.token) return;
      try {
        const res = await fetch(`${API}/api/aii/credits`, {
          headers: {'Authorization': `Bearer ${this.token}`}
        });
        
        if(!res.ok) throw new Error('Invalid token');
        
        const data = await res.json();
        document.getElementById('qt-credits').textContent = 
          data.credits.toFixed(2) + ' credits';
      } catch(e) {
        localStorage.removeItem('qt-token');
        this.setupUI();
      }
    }
  }

  // Initialize
  new Solver();
})();
