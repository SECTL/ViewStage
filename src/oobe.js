/**
 * ViewStage OOBE — 首次引导设置
 * 纯原生 JS，滑动向导
 */
import { checkForUpdate, startDownload, onProgress, offProgress } from './modules/update/update.js';

const invoke = window.__TAURI__?.core?.invoke;
const _t = (key) => window.i18n?.format_translate(key) ?? key;

// ==================== State ====================
const state = {
  step: 0,
  language: 'zh-CN',
  theme: 'com.viewstage.theme.simplify',
  penEffectMode: 'limited',
  dynamicDprEnabled: true,
  frameRateMode: 'adaptive',
  blackboardEnabled: true,
  restoreLastDoc: true,
  memCleanEnabled: true,
  cameraDeviceId: '',
  cameraWidth: null,
  cameraHeight: null,
  defaultRotation: 0,
  cameraStream: null,
  cameraDevices: [],
  importedSettings: null,
  updateChecked: false,
  updateResult: null,
}

let _updateTimeout = null;
let _downloadFilePath = null;

// ==================== Icons ====================
const ICONS = {
  globe: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12.0001 1.99805C17.5238 1.99805 22.0016 6.47589 22.0016 11.9996C22.0016 17.5233 17.5238 22.0011 12.0001 22.0011C6.47638 22.0011 1.99854 17.5233 1.99854 11.9996C1.99854 6.47589 6.47638 1.99805 12.0001 1.99805ZM14.939 16.4993H9.06118C9.71322 18.9135 10.8466 20.5011 12.0001 20.5011C13.1536 20.5011 14.2869 18.9135 14.939 16.4993ZM7.5084 16.4999L4.78591 16.4998C5.74425 18.0328 7.1777 19.2384 8.88008 19.9104C8.3578 19.0906 7.92681 18.0643 7.60981 16.8949L7.5084 16.4999ZM19.2143 16.4998L16.4918 16.4999C16.168 17.8337 15.7004 18.9995 15.119 19.9104C16.716 19.2804 18.0757 18.1814 19.0291 16.7833L19.2143 16.4998ZM7.09351 9.99895H3.7359L3.73115 10.0162C3.57906 10.6525 3.49854 11.3166 3.49854 11.9996C3.49854 13.0558 3.69112 14.0669 4.0431 14.9999L7.21626 14.9995C7.07396 14.0504 6.99854 13.0422 6.99854 11.9996C6.99854 11.3156 7.031 10.6464 7.09351 9.99895ZM15.397 9.99901H8.60316C8.53514 10.6393 8.49853 11.309 8.49853 11.9996C8.49853 13.0591 8.58468 14.0694 8.73827 14.9997H15.2619C15.4155 14.0694 15.5016 13.0591 15.5016 11.9996C15.5016 11.309 15.465 10.6393 15.397 9.99901ZM20.2647 9.99811L16.9067 9.99897C16.9692 10.6464 17.0016 11.3156 17.0016 11.9996C17.0016 13.0422 16.9262 14.0504 16.7839 14.9995L19.9571 14.9999C20.309 14.0669 20.5016 13.0558 20.5016 11.9996C20.5016 11.3102 20.4196 10.64 20.2647 9.99811ZM8.88114 4.08875L8.85823 4.09747C6.81092 4.91218 5.1549 6.49949 4.25023 8.49935L7.29835 8.49972C7.61171 6.74693 8.15855 5.221 8.88114 4.08875ZM12.0001 3.49805L11.8844 3.50335C10.619 3.6191 9.39651 5.62107 8.8288 8.4993H15.1714C14.6052 5.62914 13.388 3.63033 12.1264 3.50436L12.0001 3.49805ZM15.1201 4.08881L15.2269 4.2629C15.8961 5.37537 16.4043 6.83525 16.7018 8.49972L19.7499 8.49935C18.8853 6.58795 17.3343 5.05341 15.4113 4.21008L15.1201 4.08881Z" fill="currentColor"/></svg>',
  checkmark: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4.53033 12.9697C4.23744 12.6768 3.76256 12.6768 3.46967 12.9697C3.17678 13.2626 3.17678 13.7374 3.46967 14.0303L7.96967 18.5303C8.26256 18.8232 8.73744 18.8232 9.03033 18.5303L20.0303 7.53033C20.3232 7.23744 20.3232 6.76256 20.0303 6.46967C19.7374 6.17678 19.2626 6.17678 18.9697 6.46967L8.5 16.9393L4.53033 12.9697Z" fill="currentColor"/></svg>',
  paintBrush: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M5.75 2C5.33579 2 5 2.33579 5 2.75V14.2505C5 15.4932 6.00736 16.5005 7.25 16.5005H9.49976V19.5C9.49976 20.8807 10.619 22 11.9998 22C13.3805 22 14.4998 20.8807 14.4998 19.5V16.5005H16.75C17.9926 16.5005 19 15.4932 19 14.2505V2.75C19 2.33579 18.6642 2 18.25 2H5.75ZM6.5 11.0003V3.5H12.4998V5.25154C12.4998 5.66576 12.8355 6.00154 13.2498 6.00154C13.664 6.00154 13.9998 5.66576 13.9998 5.25154V3.5H14.9998V6.25112C14.9998 6.66534 15.3355 7.00112 15.7498 7.00112C16.164 7.00112 16.4998 6.66534 16.4998 6.25112V3.5H17.5V11.0003H6.5ZM6.5 14.2505V12.5003H17.5V14.2505C17.5 14.6647 17.1642 15.0005 16.75 15.0005H13.7498C13.3355 15.0005 12.9998 15.3363 12.9998 15.7505V19.5C12.9998 20.0523 12.552 20.5 11.9998 20.5C11.4475 20.5 10.9998 20.0523 10.9998 19.5V15.7505C10.9998 15.3363 10.664 15.0005 10.2498 15.0005H7.25C6.83579 15.0005 6.5 14.6647 6.5 14.2505Z" fill="currentColor"/></svg>',
  weatherMoon: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M20.0258 17.0014C17.2639 21.7851 11.1471 23.4241 6.3634 20.6622C5.06068 19.9101 3.964 18.8926 3.12872 17.6797C2.84945 17.2741 3.0301 16.7141 3.49369 16.5482C7.26112 15.1997 9.27892 13.6372 10.4498 11.4021C11.6825 9.04908 12.001 6.47162 11.1387 2.93862C11.0195 2.45008 11.4053 1.98492 11.9075 2.01186C13.4645 2.09539 14.9856 2.54263 16.3649 3.33903C21.1486 6.10088 22.7876 12.2177 20.0258 17.0014ZM11.7785 12.0981C10.5272 14.4867 8.46706 16.1972 4.96104 17.597C5.5693 18.2929 6.29275 18.8894 7.1134 19.3632C11.1796 21.7108 16.3791 20.3176 18.7267 16.2514C21.0744 12.1852 19.6812 6.98571 15.6149 4.63807C14.7379 4.1317 13.7951 3.79168 12.8228 3.62253C13.4699 7.00652 13.0525 9.66622 11.7785 12.0981Z" fill="currentColor"/></svg>',
  weatherSunny: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2C12.4142 2 12.75 2.33579 12.75 2.75V4.25C12.75 4.66421 12.4142 5 12 5C11.5858 5 11.25 4.66421 11.25 4.25V2.75C11.25 2.33579 11.5858 2 12 2ZM12 17C14.7614 17 17 14.7614 17 12C17 9.23858 14.7614 7 12 7C9.23858 7 7 9.23858 7 12C7 14.7614 9.23858 17 12 17ZM12 15.5C10.067 15.5 8.5 13.933 8.5 12C8.5 10.067 10.067 8.5 12 8.5C13.933 8.5 15.5 10.067 15.5 12C15.5 13.933 13.933 15.5 12 15.5ZM21.25 12.75C21.6642 12.75 22 12.4142 22 12C22 11.5858 21.6642 11.25 21.25 11.25H19.75C19.3358 11.25 19 11.5858 19 12C19 12.4142 19.3358 12.75 19.75 12.75H21.25ZM12 19C12.4142 19 12.75 19.3358 12.75 19.75V21.25C12.75 21.6642 12.4142 22 12 22C11.5858 22 11.25 21.6642 11.25 21.25V19.75C11.25 19.3358 11.5858 19 12 19ZM4.25 12.75C4.66421 12.75 5 12.4142 5 12C5 11.5858 4.66421 11.25 4.25 11.25H2.75C2.33579 11.25 2 11.5858 2 12C2 12.4142 2.33579 12.75 2.75 12.75H4.25ZM4.21967 4.22004C4.51256 3.92715 4.98744 3.92715 5.28033 4.22004L6.78033 5.72004C7.07322 6.01294 7.07322 6.48781 6.78033 6.7807C6.48744 7.0736 6.01256 7.0736 5.71967 6.7807L4.21967 5.2807C3.92678 4.98781 3.92678 4.51294 4.21967 4.22004ZM5.28033 19.7807C4.98744 20.0736 4.51256 20.0736 4.21967 19.7807C3.92678 19.4878 3.92678 19.0129 4.21967 18.72L5.71967 17.22C6.01256 16.9271 6.48744 16.9271 6.78033 17.22C7.07322 17.5129 7.07322 17.9878 6.78033 18.2807L5.28033 19.7807ZM19.7803 4.22004C19.4874 3.92715 19.0126 3.92715 18.7197 4.22004L17.2197 5.72004C16.9268 6.01294 16.9268 6.48781 17.2197 6.7807C17.5126 7.0736 17.9874 7.0736 18.2803 6.7807L19.7803 5.2807C20.0732 4.98781 20.0732 4.51294 19.7803 4.22004ZM18.7197 19.7807C19.0126 20.0736 19.4874 20.0736 19.7803 19.7807C20.0732 19.4878 20.0732 19.0129 19.7803 18.72L18.2803 17.22C17.9874 16.9271 17.5126 16.9271 17.2197 17.22C16.9268 17.5129 16.9268 17.9878 17.2197 18.2807L18.7197 19.7807Z" fill="currentColor"/></svg>',
  checkmarkCircle: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2C17.5228 2 22 6.47715 22 12C22 17.5228 17.5228 22 12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2ZM12 3.5C7.30558 3.5 3.5 7.30558 3.5 12C3.5 16.6944 7.30558 20.5 12 20.5C16.6944 20.5 20.5 16.6944 20.5 12C20.5 7.30558 16.6944 3.5 12 3.5ZM10.75 13.4393L15.2197 8.96967C15.5126 8.67678 15.9874 8.67678 16.2803 8.96967C16.5466 9.23594 16.5708 9.6526 16.3529 9.94621L16.2803 10.0303L11.2803 15.0303C11.0141 15.2966 10.5974 15.3208 10.3038 15.1029L10.2197 15.0303L7.71967 12.5303C7.42678 12.2374 7.42678 11.7626 7.71967 11.4697C7.98594 11.2034 8.4026 11.1792 8.69621 11.3971L8.78033 11.4697L10.75 13.4393L15.2197 8.96967L10.75 13.4393Z" fill="currentColor"/></svg>',
  dismiss: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4.39705 4.55379L4.46967 4.46967C4.73594 4.2034 5.1526 4.1792 5.44621 4.39705L5.53033 4.46967L12 10.939L18.4697 4.46967C18.7626 4.17678 19.2374 4.17678 19.5303 4.46967C19.8232 4.76256 19.8232 5.23744 19.5303 5.53033L13.061 12L19.5303 18.4697C19.7966 18.7359 19.8208 19.1526 19.6029 19.4462L19.5303 19.5303C19.2641 19.7966 18.8474 19.8208 18.5538 19.6029L18.4697 19.5303L12 13.061L5.53033 19.5303C5.23744 19.8232 4.76256 19.8232 4.46967 19.5303C4.17678 19.2374 4.17678 18.7626 4.46967 18.4697L10.939 12L4.46967 5.53033C4.2034 5.26406 4.1792 4.8474 4.39705 4.55379L4.46967 4.46967L4.39705 4.55379Z" fill="currentColor"/></svg>',
};

const STEPS = [
  { id: 'language' },
  { id: 'theme' },
  { id: 'importConfig' },
  { id: 'performance' },
  { id: 'drawing' },
  { id: 'cameraLoading' },
  { id: 'camera' },
  { id: 'defaultApps' },
  { id: 'checkUpdate' },
  { id: 'complete' },
  { id: 'installing' },
]

const CAMERA_STEP = 6
const PEN_COLORS = [
  { r: 239, g: 68, b: 68 },
  { r: 249, g: 115, b: 22 },
  { r: 234, g: 179, b: 8 },
  { r: 34, g: 197, b: 94 },
  { r: 6, g: 182, b: 212 },
  { r: 59, g: 130, b: 246 },
  { r: 99, g: 102, b: 241 },
  { r: 168, g: 85, b: 247 },
  { r: 236, g: 72, b: 153 },
  { r: 244, g: 63, b: 94 },
  { r: 20, g: 184, b: 166 },
  { r: 100, g: 116, b: 139 },
  { r: 30, g: 41, b: 59 },
  { r: 0, g: 0, b: 0 },
  { r: 255, g: 255, b: 255 },
]

// ==================== DOM Helpers ====================
const $ = (sel, ctx) => (ctx || document).querySelector(sel);
const $$ = (sel, ctx) => Array.from((ctx || document).querySelectorAll(sel));

function html(strings, ...vals) {
  return strings.reduce((acc, s, i) => acc + s + (vals[i] ?? ''), '');
}

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith('on')) e.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === 'className') e.className = v;
    else if (v !== undefined && v !== null) e.setAttribute(k, v);
  }
  for (const c of children) e.append(c);
  return e;
}

// ==================== Render ====================
function render() {
  const app = document.getElementById('app');
  app.innerHTML = '';
  app.append(renderShell());
  renderPanels();
  updateUI();
}

function renderShell() {
  const frag = document.createDocumentFragment();
  // Progress
  frag.append(el('div', { className: 'progress-track' },
    el('div', { className: 'progress-fill', id: 'progressFill' })
  ));
  // Dots
  const dots = el('div', { className: 'step-dots', id: 'stepDots' });
  STEPS.forEach((_, i) => dots.append(el('div', { className: 'step-dot', 'data-index': i })));
  frag.append(dots);
  // Close
  frag.append(el('div', { className: 'close-btn', id: 'closeBtn' }, '×'));
  // Slide viewport
  const track = el('div', { className: 'slide-track', id: 'slideTrack' });
  const viewport = el('div', { className: 'slide-viewport' });
  viewport.append(track);
  viewport.append(el('div', { className: 'nav-bar', id: 'navBar' }));
  frag.append(viewport);
  return frag;
}

function renderPanels() {
  const track = document.getElementById('slideTrack');
  STEPS.forEach((_, i) => {
    track.append(el('div', { className: 'slide-panel', 'data-step': i }));
  });
  renderStepContent(0);
}

function renderStepContent(stepIndex, direction) {
  const panel = $(`[data-step="${stepIndex}"]`);
  if (!panel) return;
  const tpl = STEP_TPL[stepIndex];
  panel.innerHTML = tpl ? tpl() : '';
  if (stepIndex === 0) setupLanguage();
  else if (stepIndex === 1) setupTheme();
  else if (stepIndex === 2) setupImportConfig();
  else if (stepIndex === 3) setupPerformance();
  else if (stepIndex === 4) setupDrawing();
  else if (stepIndex === 5) setupPerformancePlaceholder();
  else if (stepIndex === 6) setupCamera();
  else if (stepIndex === 7) setupDefaultApps();
  else if (stepIndex === 8) setupCheckUpdate();
  else if (stepIndex === 9) setupComplete();
  else if (stepIndex === 10) setupInstalling();
}

function updateUI() {
  const pct = ((state.step + 1) / STEPS.length) * 100;
  const fill = document.getElementById('progressFill');
  if (fill) fill.style.width = pct + '%';

  $$('.step-dot').forEach((d, i) => {
    d.className = 'step-dot';
    if (i === state.step) d.classList.add('active');
    else if (i < state.step) d.classList.add('done');
  });

  const track = document.getElementById('slideTrack');
  if (track) track.style.transform = `translateX(-${state.step * 100}vw)`;

  renderNav();
}

function renderNav() {
  const nav = document.getElementById('navBar');
  if (!nav) return;
  if (state.step === 5 || state.step === 8 || state.step === 9 || state.step === 10) {
    nav.style.display = 'none';
    return;
  }
  nav.style.display = '';
  const isFirst = state.step === 0;
  const isImport = state.step === 2;

  nav.innerHTML = html`
    <div class="nav-left">
      ${!isFirst ? `<button class="btn btn-ghost" id="btnBack">← ${_t('common.back')}</button>` : ''}
    </div>
    <div class="nav-right">
      <button class="btn btn-primary" id="btnNext">${isImport ? _t('oobe.importSkip') : _t('common.next')} →</button>
    </div>
  `;

  const backBtn = document.getElementById('btnBack');
  const nextBtn = document.getElementById('btnNext');

  if (backBtn) backBtn.addEventListener('click', prevStep);
  if (nextBtn) nextBtn.addEventListener('click', nextStep);
}

// ==================== Navigation ====================
function nextStep() {
  if (!validateStep(state.step)) return;
  if (state.step === CAMERA_STEP) {
    saveCameraSettings();
    doTransition(state.step + 1, 'forward');
  } else {
    gatherStepData(state.step);
    doTransition(state.step + 1, 'forward');
  }
}

function prevStep() {
  const target = state.step === CAMERA_STEP ? state.step - 2 : state.step - 1;
  doTransition(target);
}

function doTransition(target) {
  if (_updateTimeout) {
    clearTimeout(_updateTimeout);
    _updateTimeout = null;
  }
  const oldStep = state.step;
  state.step = target;

  const panel = $(`[data-step="${target}"]`);
  if (panel && !panel.hasChildNodes()) renderStepContent(target);

  if (target === CAMERA_STEP && oldStep !== CAMERA_STEP) initCamera();
  if (oldStep === CAMERA_STEP && target !== CAMERA_STEP) stopCamera();

  if (target === 5) setupCameraLoading();

  updateUI();
}

function validateStep(step) {
  if (step === 0) {
    const sel = $('#languageSelect .lang-item.selected');
    if (!sel) return false;
  }
  return true;
}

function gatherStepData(step) {
  if (step === 0) {
    const sel = $('#languageSelect .lang-item.selected');
    if (sel) state.language = sel.dataset.value;
  }
  if (step === 1) {
    const panel = document.querySelector('[data-step="1"]');
    const sel = panel?.querySelector('.card.selected[data-theme]');
    if (sel) state.theme = sel.dataset.theme;
  }
  if (step === 3) {
    const memToggle = document.getElementById('memCleanToggle');
    state.memCleanEnabled = memToggle ? memToggle.checked : true;
    const frGroup = document.getElementById('frameRateModeGroup');
    const frActive = frGroup?.querySelector('.option-btn.active');
    if (frActive) state.frameRateMode = frActive.dataset.value;
  }
  if (step === 4) {
    const dprToggle = document.getElementById('dynamicDprToggle');
    state.dynamicDprEnabled = dprToggle ? dprToggle.checked : true;
    const bbToggle = document.getElementById('blackboardEnabledToggle');
    state.blackboardEnabled = bbToggle ? bbToggle.checked : true;
    const group = document.getElementById('penEffectGroup');
    const active = group?.querySelector('.option-btn.active');
    if (active) state.penEffectMode = active.dataset.value;
  }
  if (step === 7) {
    const restoreToggle = document.getElementById('restoreLastDocToggle');
    state.restoreLastDoc = restoreToggle ? restoreToggle.checked : true;
  }
}

function saveCameraSettings() {
  const devOpt = $('#cameraSelect .select-option.selected');
  if (devOpt) state.cameraDeviceId = devOpt.dataset.value;
  const resOpt = $('#cameraResolutionSelect .select-option.selected');
  if (resOpt) {
    state.cameraWidth = parseInt(resOpt.dataset.width);
    state.cameraHeight = parseInt(resOpt.dataset.height);
  }
  const rotOpt = $('#defaultRotationSelect .select-option.selected');
  if (rotOpt) state.defaultRotation = parseInt(rotOpt.dataset.value);
}

// ==================== Step Templates ====================
const STEP_TPL = [
  // 0 — Language
  () => html`
    <div class="step-icon">${ICONS.globe}</div>
    <div class="step-title">${_t('oobe.selectLanguage')}</div>
    <div class="step-subtitle">${_t('oobe.selectLanguageDesc')}</div>
    <div class="lang-list" id="languageSelect">
      ${[
        { v: 'zh-CN', label: '简体中文' },
        { v: 'zh-TW', label: '繁體中文' },
        { v: 'en-US', label: 'English' },
        { v: 'ja-JP', label: '日本語' },
        { v: 'ko-KR', label: '한국어' },
        { v: 'fr-FR', label: 'Français' },
        { v: 'de-DE', label: 'Deutsch' },
        { v: 'es-ES', label: 'Español' },
        { v: 'ru-RU', label: 'Русский' },
      ].map(o => html`
        <div class="lang-item ${o.v === state.language ? 'selected' : ''}" data-value="${o.v}">
          <span class="lang-label">${o.label}</span>
          <span class="lang-check">${ICONS.checkmark}</span>
        </div>
      `).join('')}
    </div>
  `,
  // 1 — Theme
  () => html`
    <div class="step-icon">${ICONS.paintBrush}</div>
    <div class="step-title">${_t('oobe.selectTheme')}</div>
    <div class="step-subtitle">${_t('oobe.selectThemeDesc')}</div>
    <div class="card-grid">
      ${[
        { v: 'com.viewstage.theme.dark', label: _t('settings.themeDark'), icon: ICONS.weatherMoon },
        { v: 'com.viewstage.theme.simplify', label: _t('settings.themeSimplify'), icon: ICONS.weatherSunny },
      ].map(o => html`
        <div class="card ${o.v === state.theme ? 'selected' : ''}" data-theme="${o.v}">
          <div class="card-icon">${o.icon}</div>
          <div class="card-title">${o.label}</div>
        </div>
      `).join('')}
    </div>
  `,
  // 2 — Import Config
  () => html`
    <div class="step-icon"></div>
    <div class="step-title">${_t('oobe.importConfig')}</div>
    <div class="step-subtitle">${_t('oobe.importConfigDesc')}</div>
    <div style="display:flex;flex-direction:column;align-items:center;gap:16px">
      <button class="btn btn-primary" id="importConfigBtn">${_t('oobe.importConfigBtn')}</button>
      <div id="importStatus" class="import-status"></div>
    </div>
  `,
  // 3 — Performance
  () => html`
    <div class="step-icon"></div>
    <div class="step-title">${_t('oobe.performance')}</div>
    <div class="step-subtitle">${_t('oobe.performanceDesc')}</div>
    <div class="toggle-row">
      <div>
        <div class="toggle-label">${_t('settings.memreductClean')}</div>
        <div class="toggle-desc">${_t('oobe.memCleanAdminHint')}</div>
      </div>
      <label class="toggle-switch">
        <input type="checkbox" id="memCleanToggle" ${state.memCleanEnabled ? 'checked' : ''}>
        <span class="toggle-slider"></span>
      </label>
    </div>
    <div class="divider"></div>
    <div class="setting-row">
      <div class="setting-row-label">${_t('settings.frameRateMode')}</div>
      <div class="option-group" id="frameRateModeGroup">
        ${['low', 'adaptive', 'high'].map(v => html`
          <button class="option-btn${state.frameRateMode === v ? ' active' : ''}" data-value="${v}">
            ${_t('settings.frameRate' + v.charAt(0).toUpperCase() + v.slice(1))}
          </button>
        `).join('')}
      </div>
    </div>
  `,
  // 4 — Drawing
  () => html`
    <div class="step-icon"></div>
    <div class="step-title">${_t('oobe.drawingSettings')}</div>
    <div class="step-subtitle">${_t('oobe.drawingSettingsDesc')}</div>
    <div class="toggle-row">
      <div>
        <div class="toggle-label">${_t('settings.dynamicResolution')}</div>
        <div class="toggle-desc">${_t('settings.dynamicResolutionHint')}</div>
      </div>
      <label class="toggle-switch">
        <input type="checkbox" id="dynamicDprToggle" ${state.dynamicDprEnabled ? 'checked' : ''}>
        <span class="toggle-slider"></span>
      </label>
    </div>
    <div class="toggle-row">
      <div>
        <div class="toggle-label">${_t('settings.blackboardEnabled')}</div>
        <div class="toggle-desc">${_t('settings.blackboardEnabledHint')}</div>
      </div>
      <label class="toggle-switch">
        <input type="checkbox" id="blackboardEnabledToggle" ${state.blackboardEnabled ? 'checked' : ''}>
        <span class="toggle-slider"></span>
      </label>
    </div>
    <div class="divider"></div>
    <div class="setting-row">
      <div class="setting-row-label">${_t('settings.penEffect')}</div>
      <div class="option-group" id="penEffectGroup">
        ${['off', 'limited', 'full'].map(v => html`
          <button class="option-btn${state.penEffectMode === v ? ' active' : ''}" data-value="${v}">
            ${_t('settings.penEffect' + v.charAt(0).toUpperCase() + v.slice(1))}
          </button>
        `).join('')}
      </div>
    </div>
  `,
  // 5 — Camera Loading
  () => html`
    <div class="camera-loading">
      <div class="spinner"></div>
      <div class="camera-loading-text">${_t('oobe.cameraLoading')}</div>
    </div>
  `,
  // 6 — Camera
  () => html`
    <div class="step-icon"></div>
    <div class="step-title">${_t('oobe.cameraSetup')}</div>
    <div class="step-subtitle">${_t('oobe.cameraSetupDesc')}</div>
    <div class="camera-section">
      <div class="camera-viewport">
        <video id="cameraPreview" autoplay playsinline muted></video>
        <div class="camera-placeholder" id="cameraPlaceholder">
          <div class="icon"></div>
          <span>${_t('oobe.cameraPreviewLoading')}</span>
        </div>
      </div>
      <div class="camera-selects">
        <div class="select-row">
          <label>${_t('settings.defaultCamera')}</label>
          <div class="custom-select" id="cameraSelect">
            <div class="select-selected" id="cameraSelected">${_t('common.loading')}</div>
            <div class="select-options" id="cameraOptions"></div>
          </div>
        </div>
        <div class="select-row">
          <label>${_t('settings.cameraResolution')}</label>
          <div class="custom-select" id="cameraResolutionSelect">
            <div class="select-selected" id="cameraResolutionSelected">${_t('common.loading')}</div>
            <div class="select-options" id="cameraResolutionOptions"></div>
          </div>
        </div>
        <div class="select-row">
          <label>${_t('settings.defaultRotation')}</label>
          <div class="custom-select" id="defaultRotationSelect">
            <div class="select-selected" id="defaultRotationSelected">0° ${_t('settings.rotationNone')}</div>
            <div class="select-options" id="defaultRotationOptions"></div>
          </div>
        </div>
      </div>
    </div>
  `,
  // 7 — Default Apps
  () => html`
    <div class="step-icon"></div>
    <div class="step-title">${_t('oobe.defaultApps')}</div>
    <div class="step-subtitle">${_t('oobe.defaultAppsDesc')}</div>
    <div class="default-apps">
      <div class="app-row">
        <span class="app-row-icon"></span>
        <div class="app-row-info">
          <div class="app-row-name">PDF</div>
          <div class="app-row-desc">${_t('settings.pdfDefaultOpen')}</div>
        </div>
        <span id="pdfDefaultStatus" class="app-status"></span>
        <button class="btn btn-ghost" id="btnSetPdf">${_t('settings.setDefault')}</button>
      </div>
      <div class="app-row">
        <span class="app-row-icon"></span>
        <div class="app-row-info">
          <div class="app-row-name">Word</div>
          <div class="app-row-desc">${_t('settings.wordDefaultOpen')}</div>
        </div>
        <span id="wordDefaultStatus" class="app-status"></span>
        <button class="btn btn-ghost" id="btnSetWord">${_t('settings.setDefault')}</button>
      </div>
    </div>
    <div class="divider"></div>
    <div class="toggle-row">
      <div>
        <div class="toggle-label">${_t('settings.restoreLastDoc')}</div>
        <div class="toggle-desc">${_t('settings.restoreLastDocHint')}</div>
      </div>
      <label class="toggle-switch">
        <input type="checkbox" id="restoreLastDocToggle" ${state.restoreLastDoc ? 'checked' : ''}>
        <span class="toggle-slider"></span>
      </label>
    </div>
  `,
  // 8 — Check Update
  () => html`
    <div class="step-icon"></div>
    <div class="step-title">${_t('oobe.checkUpdate')}</div>
    <div class="step-subtitle">${_t('oobe.checkUpdateDesc')}</div>
    <div class="update-banner" id="updateBanner" style="display:none"></div>
    <div class="release-notes" id="updateNotes" style="display:none"></div>
    <div class="update-actions" id="updateActions">
      <button class="btn btn-primary" id="btnUpdateDownload" style="display:none">${_t('oobe.updateNow')}</button>
      <button class="btn btn-ghost" id="btnUpdateSkip" style="display:none">${_t('oobe.updateLater')}</button>
    </div>
    <div class="update-progress" id="updateProgress" style="display:none">
      <div class="progress-bar"><div class="progress-fill" id="updateProgressBar"></div></div>
      <div class="progress-label" id="updateProgressText">0%</div>
    </div>
    <div id="updateStatus">
      <div class="spinner"></div>
      <div class="update-text">${_t('oobe.checkingUpdate')}</div>
    </div>
  `,
  // 9 — Complete
  () => html`
    <div class="checkmark"></div>
    <div class="step-title">${_t('oobe.setupComplete')}</div>
    <div class="step-subtitle">${_t('oobe.configSaved')}</div>
    <div style="margin-top:16px">
      <button class="btn btn-success" id="btnFinishStep">${_t('oobe.restartApp')}</button>
    </div>
  `,
  // 10 — Installing
  () => html`
    <div class="installing-view">
      <div class="spinner"></div>
      <div class="installing-text" id="installingText">${_t('oobe.saving')}</div>
    </div>
  `,
]

// ==================== Step Setup Functions ====================
function setupLanguage() {
  const grid = document.getElementById('languageSelect');
  if (!grid) return;
  grid.addEventListener('click', async (e) => {
    const item = e.target.closest('.lang-item');
    if (!item) return;
    const locale = item.dataset.value;
    if (window.i18n && locale !== state.language) {
      state.language = locale;
      const scrollTop = grid.scrollTop;
      await window.i18n.load_messages(locale);
      window.i18n.render_page_texts();
      localStorage.setItem('language', locale);
      document.documentElement.lang = locale;
      renderStepContent(0);
      const newGrid = document.getElementById('languageSelect');
      if (newGrid) newGrid.scrollTop = scrollTop;
    }
  });
}

function setupTheme() {
  const panel = document.querySelector('[data-step="1"]');
  if (!panel) return;
  const grid = panel.querySelector('.card-grid');
  if (!grid) return;
  grid.addEventListener('click', async (e) => {
    const card = e.target.closest('.card');
    if (!card) return;
    $$('.card', grid).forEach(c => c.classList.remove('selected'));
    card.classList.add('selected');
    const themeId = card.dataset.theme;
    if (themeId && window.ThemeManager) {
      state.theme = themeId;
      await window.ThemeManager.theme_update_active(themeId);
    }
  });
}

function setupImportConfig() {
  document.getElementById('importConfigBtn')?.addEventListener('click', importConfig);
}

function setupPerformance() {
  const frGroup = document.getElementById('frameRateModeGroup');
  if (frGroup) {
    frGroup.dataset.active = state.frameRateMode;
    frGroup.querySelectorAll('.option-btn').forEach(btn => {
      if (btn.dataset.value === state.frameRateMode) btn.classList.add('active');
    });
    frGroup.addEventListener('click', (e) => {
      const btn = e.target.closest('.option-btn');
      if (!btn) return;
      const mode = btn.dataset.value;
      state.frameRateMode = mode;
      frGroup.dataset.active = mode;
      frGroup.querySelectorAll('.option-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  }
  const toggle = document.getElementById('memCleanToggle');
  if (toggle) {
    toggle.addEventListener('change', (e) => {
      state.memCleanEnabled = e.target.checked;
    });
    (async () => {
      if (!invoke) return;
      try {
        const exists = await invoke('memreduct_check_skipuac');
        if (!exists) {
          state.memCleanEnabled = false;
          toggle.checked = false;
        }
      } catch (_) {
        state.memCleanEnabled = false;
        toggle.checked = false;
      }
    })();
  }
}

function setupDrawing() {
  const penGroup = document.getElementById('penEffectGroup');
  if (penGroup) {
    penGroup.dataset.active = state.penEffectMode;
    penGroup.querySelectorAll('.option-btn').forEach(btn => {
      if (btn.dataset.value === state.penEffectMode) btn.classList.add('active');
    });
    penGroup.addEventListener('click', (e) => {
      const btn = e.target.closest('.option-btn');
      if (!btn) return;
      const mode = btn.dataset.value;
      state.penEffectMode = mode;
      penGroup.dataset.active = mode;
      penGroup.querySelectorAll('.option-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  }
  document.getElementById('dynamicDprToggle')?.addEventListener('change', (e) => {
    state.dynamicDprEnabled = e.target.checked;
  });
  document.getElementById('blackboardEnabledToggle')?.addEventListener('change', (e) => {
    state.blackboardEnabled = e.target.checked;
  });
}

function setupPerformancePlaceholder() {
  // No-op; setupCameraLoading is called from doTransition
}

function setupCameraLoading() {
  if (state.cameraStream) {
    doTransition(CAMERA_STEP, 'forward');
    return;
  }

  const loadingStep = state.step;
  Promise.race([
    navigator.mediaDevices.getUserMedia({ video: true }),
    new Promise((_, rej) => setTimeout(() => rej(new Error('Timeout')), 5000)),
  ])
    .then(stream => {
      if (state.step !== loadingStep) {
        stream.getTracks().forEach(t => t.stop());
        return;
      }
      state.cameraStream = stream;
      doTransition(CAMERA_STEP, 'forward');
    })
    .catch(() => {
      if (state.step !== loadingStep) return;
      doTransition(CAMERA_STEP, 'forward');
    });
}

function setupCamera() {
  initCustomSelects();
  initDefaultRotationSelect();
}

function setupDefaultApps() {
  checkAssociation('pdf', 'pdfDefaultStatus');
  checkAssociation('word', 'wordDefaultStatus');

  document.getElementById('btnSetPdf')?.addEventListener('click', async () => {
    if (!invoke) return;
    try {
      await invoke('filetype_set_icons');
      checkAssociation('pdf', 'pdfDefaultStatus');
    } catch (_) {}
  });

  document.getElementById('btnSetWord')?.addEventListener('click', async () => {
    if (!invoke) return;
    try {
      await invoke('filetype_set_icons');
      checkAssociation('word', 'wordDefaultStatus');
    } catch (_) {}
  });
}

async function checkAssociation(ext, statusElId) {
  const statusEl = document.getElementById(statusElId);
  if (!statusEl || !invoke) return;
  try {
    const cmd = ext === 'pdf' ? 'filetype_validate_pdf_default' : 'filetype_validate_word_default';
    const ok = await invoke(cmd);
    statusEl.innerHTML = ok ? ICONS.checkmarkCircle : ICONS.dismiss;
    statusEl.className = 'app-status ' + (ok ? 'associated' : 'not-associated');
  } catch (_) {
    statusEl.innerHTML = ICONS.dismiss;
    statusEl.className = 'app-status not-associated';
  }
}

function setupComplete() {
  document.getElementById('btnFinishStep')?.addEventListener('click', () => doTransition(10));
}

async function setupInstalling() {
  try {
    await performFinalSave();
    if (_downloadFilePath) {
      await invoke('update_install_release', { filePath: _downloadFilePath });
    } else {
      await invoke('oobe_submit_complete');
    }
    // 遥测上报：OOBE 完成
    import('./modules/telemetry/telemetry.js').then(m => {
      m.reportOnline();
    }).catch(e => {
      console.warn('[oobe] telemetry report failed:', e);
    });
  } catch (err) {
    console.error('Setup failed:', err);
    if (_downloadFilePath) {
      doTransition(8);
    } else {
      doTransition(9);
    }
  }
}

async function setupCheckUpdate() {
  const statusEl = document.getElementById('updateStatus');

  if (state.updateChecked) {
    showUpdateResult(state.updateResult);
  } else {
    if (statusEl) statusEl.innerHTML = `<div class="spinner"></div><div class="update-text">${_t('oobe.checkingUpdate')}</div>`;
    _checkForUpdate();
  }
}

async function showUpdateResult(result) {
  const statusEl = document.getElementById('updateStatus');
  const bannerEl = document.getElementById('updateBanner');
  const downloadBtn = document.getElementById('btnUpdateDownload');
  const skipBtn = document.getElementById('btnUpdateSkip');
  if (!bannerEl || !statusEl) return;

  if (!result) {
    bannerEl.className = 'update-banner banner-error';
    bannerEl.textContent = _t('oobe.updateCheckFailed');
    bannerEl.style.display = '';
    statusEl.innerHTML = '';
    const notesEl = document.getElementById('updateNotes');
    if (notesEl) notesEl.style.display = 'none';
    _updateTimeout = setTimeout(() => doTransition(9, 'forward'), 2000);
    return;
  }

  if (result.has_update) {
    statusEl.innerHTML = '';
    bannerEl.style.display = 'none';
    const notesEl = document.getElementById('updateNotes');
    if (notesEl) {
      if (result.release?.body) {
        notesEl.innerHTML = renderMarkdownSimple(result.release.body);
        notesEl.style.display = '';
      } else {
        notesEl.style.display = 'none';
      }
    }
    if (downloadBtn) downloadBtn.style.display = '';
    if (skipBtn) skipBtn.style.display = '';

    document.getElementById('btnUpdateSkip')?.addEventListener('click', () => doTransition(9, 'forward'));
    document.getElementById('btnUpdateDownload')?.addEventListener('click', async () => {
      if (!invoke || !result.release?.assets?.length) return;

      if (_downloadFilePath) {
        doTransition(10);
        return;
      }

      try {
        const platform = await invoke('app_fetch_platform');

        const progressEl = document.getElementById('updateProgress');
        const progressBar = document.getElementById('updateProgressBar');
        const progressText = document.getElementById('updateProgressText');
        if (downloadBtn) downloadBtn.style.display = 'none';
        if (skipBtn) skipBtn.style.display = 'none';
        if (progressEl) progressEl.style.display = '';
        if (progressBar) progressBar.style.width = '0%';
        if (progressText) progressText.textContent = '0%';

        offProgress();
        await onProgress((p) => {
          if (progressBar) progressBar.style.width = p + '%';
          if (progressText) progressText.textContent = Math.round(p) + '%';
        });

        _downloadFilePath = await startDownload(result.release, platform, '');

        offProgress();

        if (progressEl) progressEl.style.display = 'none';
        if (downloadBtn) { downloadBtn.style.display = ''; downloadBtn.disabled = false; downloadBtn.textContent = _t('oobe.installNow') || 'Install Now'; }
        if (skipBtn) skipBtn.style.display = 'none';
      } catch (err) {
        console.error('Download failed:', err);
        if (downloadBtn) { downloadBtn.style.display = ''; downloadBtn.disabled = false; downloadBtn.textContent = _t('oobe.updateNow'); }
        if (skipBtn) skipBtn.style.display = '';
        const progEl = document.getElementById('updateProgress');
        if (progEl) progEl.style.display = 'none';
      }
    });
  } else {
    statusEl.innerHTML = '';
    bannerEl.className = 'update-banner banner-latest';
    bannerEl.textContent = _t('oobe.updateUpToDate');
    bannerEl.style.display = '';
    const notesEl = document.getElementById('updateNotes');
    if (notesEl) notesEl.style.display = 'none';
    await new Promise(r => setTimeout(r, 1200));
    doTransition(9, 'forward');
  }
}

async function _checkForUpdate() {
  const statusEl = document.getElementById('updateStatus');
  const notesEl = document.getElementById('updateNotes');
  if (notesEl) notesEl.style.display = 'none';

  try {
    const { result } = await checkForUpdate();
    state.updateChecked = true;
    state.updateResult = result;
    showUpdateResult(result);
  } catch (err) {
    console.warn('Update check failed:', err);
    state.updateChecked = true;
    state.updateResult = null;
    if (statusEl) statusEl.innerHTML = '';
    const bannerEl = document.getElementById('updateBanner');
    if (bannerEl) {
      bannerEl.className = 'update-banner banner-error';
      bannerEl.textContent = _t('oobe.updateCheckFailed');
      bannerEl.style.display = '';
    }
    _updateTimeout = setTimeout(() => doTransition(9, 'forward'), 2000);
  }
}

async function performFinalSave() {
  gatherStepData(state.step);
  saveCameraSettings();

  if (state.memCleanEnabled && invoke) {
    try {
      const exists = await invoke('memreduct_check_skipuac');
      if (!exists) {
        await invoke('memreduct_setup');
      }
    } catch (err) {
      console.warn('MemClean scheduled task setup failed, disabling:', err);
      state.memCleanEnabled = false;
    }
  }

  const finalSettings = mergeSettings();
  await invoke('device_detect_all');
  await invoke('settings_save_all', { settings: finalSettings });
}

// ==================== Camera ====================
async function initCamera() {
  const video = document.getElementById('cameraPreview');
  const placeholder = document.getElementById('cameraPlaceholder');
  const camSelect = document.getElementById('cameraSelect');
  const camOptions = document.getElementById('cameraOptions');
  const camSelected = document.getElementById('cameraSelected');
  const resSelected = document.getElementById('cameraResolutionSelected');

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    state.cameraDevices = devices.filter(d => d.kind === 'videoinput');

    if (state.cameraDevices.length === 0) {
      camSelected.textContent = _t('settings.noCameraDetected');
      resSelected.textContent = '-';
      if (camSelect) camSelect.classList.add('disabled');
      return;
    }

    let stream = state.cameraStream;
    if (!stream) {
      stream = await Promise.race([
        navigator.mediaDevices.getUserMedia({ video: true }),
        new Promise((_, rej) => setTimeout(() => rej(new Error('Timeout')), 5000)),
      ]);
      state.cameraStream = stream;
    }

    video.srcObject = stream;
    video.classList.add('active');
    placeholder.classList.add('hidden');

    camOptions.innerHTML = '';
    state.cameraDevices.forEach((d, i) => {
      const opt = document.createElement('div');
      opt.className = 'select-option' + (i === 0 ? ' selected' : '');
      opt.dataset.value = d.deviceId;
      opt.textContent = d.label || `${_t('camera.camera')} ${i + 1}`;
      camOptions.appendChild(opt);
    });

    camSelected.textContent = state.cameraDevices[0].label || `${_t('camera.camera')} 1`;
    await initResolutionSelect(stream);

    setupCustomSelects();
  } catch (err) {
    console.error('Camera init failed:', err);
    if (err.name === 'NotAllowedError') {
      camSelected.textContent = _t('settings.noCameraPermission');
    } else {
      camSelected.textContent = _t('settings.noCameraDetected');
    }
    resSelected.textContent = '-';
    if (camSelect) camSelect.classList.add('disabled');
  }
}

function stopCamera() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach(t => t.stop());
    state.cameraStream = null;
  }
}

async function initResolutionSelect(stream) {
  const options = document.getElementById('cameraResolutionOptions');
  const selected = document.getElementById('cameraResolutionSelected');
  if (!options || !selected) return;

  const common = [
    { w: 640, h: 480, label: '640×480 (VGA)' },
    { w: 800, h: 600, label: '800×600 (SVGA)' },
    { w: 1280, h: 720, label: '1280×720 (720p)' },
    { w: 1280, h: 960, label: '1280×960' },
    { w: 1600, h: 1200, label: '1600×1200' },
    { w: 1920, h: 1080, label: '1920×1080 (1080p)' },
    { w: 2560, h: 1440, label: '2560×1440 (2K)' },
    { w: 3840, h: 2160, label: '3840×2160 (4K)' },
  ];

  const track = stream.getVideoTracks()[0];
  const caps = track?.getCapabilities();
  const maxW = caps?.width?.max || 1920;
  const maxH = caps?.height?.max || 1080;

  const filtered = common.filter(r => r.w <= maxW && r.h <= maxH);
  const hasMax = filtered.some(r => r.w === maxW && r.h === maxH);
  if (!hasMax) filtered.push({ w: maxW, h: maxH, label: `${maxW}×${maxH} (${_t('oobe.max')})` });
  filtered.sort((a, b) => (b.w * b.h) - (a.w * a.h));

  const def = filtered[0];
  const defIdx = 0;

  options.innerHTML = '';
  filtered.forEach((r, i) => {
    const opt = document.createElement('div');
    opt.className = 'select-option' + (i === defIdx ? ' selected' : '');
    opt.dataset.width = r.w;
    opt.dataset.height = r.h;
    opt.dataset.value = `${r.w}×${r.h}`;
    opt.textContent = r.label;
    options.appendChild(opt);
  });

  selected.textContent = def.label;
}

// ==================== Custom Selects ====================
function setupCustomSelects() {
  function closeOneSelect(s) {
    s.classList.remove('open');
    const opts = document.querySelector('body > .select-options[data-owner="' + s.dataset.selectId + '"]');
    if (opts) {
      opts.style.opacity = '0';
      opts.style.visibility = 'hidden';
      setTimeout(() => {
        if (!s.classList.contains('open')) {
          s.appendChild(opts);
          opts.classList.remove('up');
          opts.style.position = '';
          opts.style.left = '';
          opts.style.top = '';
          opts.style.bottom = '';
          opts.style.minWidth = '';
          opts.style.transform = '';
        }
      }, 200);
    }
  }

  function closeAllSelects() {
    $$('.custom-select.open').forEach(s => closeOneSelect(s));
  }

  document.removeEventListener('click', closeAllSelects);
  document.addEventListener('click', closeAllSelects);

  let selectId = 0;
  $$('.custom-select').forEach(select => {
    const selected = select.querySelector('.select-selected');
    if (!selected || select.dataset.selectInitialized) return;
    select.dataset.selectInitialized = 'true';
    const id = 'sel_' + (selectId++);
    select.dataset.selectId = id;
    const opts = select.querySelector('.select-options');
    if (opts) opts.dataset.owner = id;

    selected.addEventListener('click', (e) => {
      e.stopPropagation();
      $$('.custom-select.open').forEach(s => {
        if (s !== select) closeOneSelect(s);
      });
      const isOpen = select.classList.toggle('open');
      if (opts) {
        if (isOpen) {
          const rect = selected.getBoundingClientRect();
          if (opts.parentNode !== document.body) {
            document.body.appendChild(opts);
          }
          opts.style.visibility = 'hidden';
          opts.style.opacity = '1';
          void opts.offsetHeight;
          const spaceBelow = window.innerHeight - rect.bottom;
          const spaceAbove = rect.top;
          const needed = opts.scrollHeight || 180;
          const showUp = select.dataset.selectUp === 'always' || (spaceBelow < needed && spaceAbove > spaceBelow);
          opts.classList.toggle('up', showUp);
          opts.style.position = 'fixed';
          opts.style.left = rect.left + 'px';
          if (showUp) {
            opts.style.top = '';
            opts.style.bottom = (window.innerHeight - rect.top + 4) + 'px';
          } else {
            opts.style.top = (rect.bottom + 4) + 'px';
            opts.style.bottom = '';
          }
          opts.style.minWidth = rect.width + 'px';
          opts.style.opacity = '1';
          opts.style.visibility = 'visible';
          opts.style.transform = 'translateY(0)';
        } else {
          if (opts.parentNode !== select) {
            select.appendChild(opts);
          }
          opts.classList.remove('up');
          opts.style.position = '';
          opts.style.left = '';
          opts.style.top = '';
          opts.style.bottom = '';
          opts.style.minWidth = '';
          opts.style.opacity = '';
          opts.style.visibility = '';
          opts.style.transform = '';
        }
      }
    });

    opts?.addEventListener('click', async (e) => {
      const opt = e.target.closest('.select-option');
      if (!opt) return;
      selected.textContent = opt.textContent;
      $$('.select-option', opts).forEach(o => o.classList.remove('selected'));
      opt.classList.add('selected');
      closeOneSelect(select);

      if (select.id === 'cameraSelect') {
        try {
          const deviceId = opt.dataset.value;
          if (state.cameraStream) {
            state.cameraStream.getTracks().forEach(t => t.stop());
            state.cameraStream = null;
          }
          const stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: deviceId } } });
          state.cameraStream = stream;
          document.getElementById('cameraPreview').srcObject = stream;
          await initResolutionSelect(stream);
        } catch (err) {
          console.error('Camera switch failed:', err);
        }
      }

      if (select.id === 'cameraResolutionSelect') {
        applyCameraResolution();
      }

      if (select.id === 'defaultRotationSelect') {
        applyCameraRotation();
      }
    });
  });
}

function initCustomSelects() {
  setupCustomSelects();
}

function initDefaultRotationSelect() {
  const options = document.getElementById('defaultRotationOptions');
  const selected = document.getElementById('defaultRotationSelected');
  if (!options || !selected) return;
  const rotations = [
    { value: 0, label: _t('settings.rotationNone') },
    { value: 90, label: _t('settings.rotationClockwise') },
    { value: 180, label: _t('settings.rotationUpsideDown') },
    { value: 270, label: _t('settings.rotationCounterClockwise') },
  ];
  options.innerHTML = rotations.map(r => `
    <div class="select-option${state.defaultRotation === r.value ? ' selected' : ''}" data-value="${r.value}">
      ${r.value}° (${r.label})
    </div>
  `).join('');
  const active = rotations.find(r => r.value === state.defaultRotation);
  if (active) selected.textContent = `${active.value}° (${active.label})`;
}

function applyCameraResolution() {
  const resOpt = $('#cameraResolutionSelect .select-option.selected');
  if (!resOpt || !state.cameraStream) return;
  const w = parseInt(resOpt.dataset.width);
  const h = parseInt(resOpt.dataset.height);
  const track = state.cameraStream.getVideoTracks()[0];
  if (track) {
    track.applyConstraints({ width: { ideal: w }, height: { ideal: h } }).catch(() => {});
  }
}

function applyCameraRotation() {
  const rotOpt = $('#defaultRotationSelect .select-option.selected');
  if (!rotOpt) return;
  const deg = parseInt(rotOpt.dataset.value);
  state.defaultRotation = deg;
  const video = document.getElementById('cameraPreview');
  if (video) {
    video.style.transform = `rotate(${deg}deg)`;
  }
}

document.addEventListener('click', () => {
  $$('.custom-select.open').forEach(s => s.classList.remove('open'));
});

// ==================== Config Import ====================
async function importConfig() {
  try {
    const { open } = window.__TAURI__.dialog;
    const { readTextFile } = window.__TAURI__.fs;

    const filePath = await open({ filters: [{ name: 'JSON', extensions: ['json'] }] });
    if (!filePath) return;

    const jsonStr = await readTextFile(filePath);
    const settings = JSON.parse(jsonStr);

    if (!settings || typeof settings !== 'object' || !settings.language) {
      showImportStatus('error', _t('oobe.importFailed'));
      return;
    }

    state.importedSettings = settings;
    showImportStatus('success', _t('oobe.importSuccess'));
    setTimeout(() => doTransition(8), 800);
  } catch (err) {
    console.error('Import failed:', err);
    showImportStatus('error', _t('oobe.importFailed'));
  }
}

function showImportStatus(type, msg) {
  const el = document.getElementById('importStatus');
  if (!el) return;
  el.textContent = msg;
  el.className = 'import-status ' + type;
}

// ==================== Config Save ====================
function mergeSettings() {
  const base = state.importedSettings ? { ...state.importedSettings } : {
    language: 'zh-CN',
    theme: 'com.viewstage.theme.simplify',
    defaultCamera: '',
    cameraWidth: null,
    cameraHeight: null,
    penColors: PEN_COLORS.map(c => ({ r: c.r, g: c.g, b: c.b })),
  };

  return {
    ...base,
    language: state.language,
    theme: state.theme,
    penEffectMode: state.penEffectMode,
    dynamicDprEnabled: state.dynamicDprEnabled,
    frameRateMode: state.frameRateMode,
    blackboardEnabled: state.blackboardEnabled,
    restoreLastDoc: state.restoreLastDoc,
    defaultRotation: state.defaultRotation,
    memreductCleanEnabled: state.memCleanEnabled,
    defaultCamera: state.cameraDeviceId,
    cameraWidth: state.cameraWidth,
    cameraHeight: state.cameraHeight,
    penColors: PEN_COLORS,
    oobeCompleted: true,
  };
}

// ==================== Init ====================
async function init() {
  if (window.i18n) {
    await window.i18n.init_start();
  }
  if (window.ThemeManager) {
    await window.ThemeManager.init();
  }

  render();

  document.getElementById('closeBtn')?.addEventListener('click', async () => {
    if (invoke) await invoke('app_submit_exit');
  });
}

init();
