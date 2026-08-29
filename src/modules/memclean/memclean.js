let _memclean_invoke = null;
let _memclean_mask = 231;

const REGIONS = [
  { bit: 0x01, key: 'memclean.regionWorkingSet' },
  { bit: 0x02, key: 'memclean.regionSystemFileCache' },
  { bit: 0x04, key: 'memclean.regionStandbyPriority0' },
  { bit: 0x08, key: 'memclean.regionStandbyList' },
  { bit: 0x10, key: 'memclean.regionModifiedList' },
  { bit: 0x20, key: 'memclean.regionCombineMemory' },
  { bit: 0x40, key: 'memclean.regionRegistryCache' },
];

function _mqs(id) { return document.getElementById(id); }

async function _memclean_load_mask() {
  if (!_memclean_invoke) return;
  try {
    const result = await window.getSettings();
    _memclean_mask = result.settings?.memclean?.mask ?? 231;
  } catch {
    _memclean_mask = 231;
  }
  _memclean_apply_mask_to_checkboxes();
}

async function _memclean_save_mask(mask) {
  if (!_memclean_invoke) return;
  _memclean_mask = mask;
  try {
    await _memclean_invoke('settings_save_all', { settings: { memclean: { mask } } });
  } catch {
    /* silently ignore */
  }
}

function _memclean_recompute_mask() {
  let mask = 0;
  document.querySelectorAll('#memcleanRegions input[type="checkbox"]').forEach(cb => {
    if (cb.checked) mask |= parseInt(cb.dataset.bit, 10);
  });
  return mask;
}

function _memclean_apply_mask_to_checkboxes() {
  document.querySelectorAll('#memcleanRegions input[type="checkbox"]').forEach(cb => {
    cb.checked = (_memclean_mask & parseInt(cb.dataset.bit, 10)) !== 0;
  });
}

function _memclean_build_region_ui(regions) {
  const container = _mqs('memcleanRegions');
  if (!container) return;
  container.innerHTML = '';
  for (const r of regions) {
    const label = document.createElement('label');
    label.className = 'memclean-region-item';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.dataset.bit = r.bit;
    const span = document.createElement('span');
    span.textContent = window.i18n.format_translate(r.key);
    label.appendChild(cb);
    label.appendChild(span);
    container.appendChild(label);
  }
  container.addEventListener('change', () => {
    _memclean_save_mask(_memclean_recompute_mask());
  });
}

async function memclean_refresh() {
  const dot = _mqs('memcleanStatusDot');
  const text = _mqs('memcleanStatusText');
  const setupRow = _mqs('memcleanSetupRow');
  const uninstallRow = _mqs('memcleanUninstallRow');

  if (!_memclean_invoke) {
    if (dot) dot.className = 'memclean-status-dot error';
    if (text) text.textContent = window.i18n.format_translate('memclean.statusIpcUnavailable');
    return;
  }

  dot.className = 'memclean-status-dot inactive';
  text.textContent = window.i18n.format_translate('memclean.statusChecking');

  try {
    const enabled = await _memclean_invoke('memreduct_check_skipuac');

    if (enabled) {
      dot.className = 'memclean-status-dot active';
      text.textContent = window.i18n.format_translate('memclean.statusReady');
      if (setupRow) setupRow.style.display = 'none';
      if (uninstallRow) uninstallRow.style.display = '';
    } else {
      dot.className = 'memclean-status-dot inactive';
      text.textContent = window.i18n.format_translate('memclean.statusNotSetup');
      if (setupRow) setupRow.style.display = '';
      if (uninstallRow) uninstallRow.style.display = 'none';
    }
  } catch {
    dot.className = 'memclean-status-dot error';
    text.textContent = window.i18n.format_translate('memclean.statusError');
  }
}

async function _memclean_do_clean() {
  const btn = _mqs('memcleanCleanBtn');
  if (!btn || !_memclean_invoke) return;
  btn.disabled = true;
  btn.textContent = window.i18n.format_translate('memclean.cleaning');
  try {
    const mask = _memclean_recompute_mask();
    await _memclean_invoke('memreduct_clean_now', { mask });
    btn.textContent = window.i18n.format_translate('memclean.cleanDone');
    setTimeout(() => { btn.textContent = window.i18n.format_translate('memclean.cleanNow'); btn.disabled = false; }, 2500);
  } catch {
    btn.textContent = window.i18n.format_translate('memclean.cleanFailed');
    setTimeout(() => { btn.textContent = window.i18n.format_translate('memclean.cleanNow'); btn.disabled = false; }, 3000);
  }
}

async function _memclean_do_setup() {
  const btn = _mqs('memcleanSetupBtn');
  if (!btn || !_memclean_invoke) return;
  btn.disabled = true;
  btn.textContent = window.i18n.format_translate('memclean.settingUp');
  try {
    await _memclean_invoke('memreduct_setup');
    btn.textContent = window.i18n.format_translate('memclean.setupDone');
    await memclean_refresh();
    setTimeout(() => { btn.textContent = window.i18n.format_translate('memclean.setupTask'); btn.disabled = false; }, 2500);
  } catch {
    btn.textContent = window.i18n.format_translate('memclean.setupFailed');
    btn.disabled = false;
  }
}

async function _memclean_do_uninstall() {
  const btn = _mqs('memcleanUninstallBtn');
  if (!btn || !_memclean_invoke) return;
  btn.disabled = true;
  btn.textContent = window.i18n.format_translate('memclean.uninstalling');
  try {
    await _memclean_invoke('memreduct_uninstall');
    btn.textContent = window.i18n.format_translate('memclean.uninstallDone');
    await memclean_refresh();
    setTimeout(() => { btn.textContent = window.i18n.format_translate('memclean.uninstallTask'); btn.disabled = false; }, 2500);
  } catch {
    btn.textContent = window.i18n.format_translate('memclean.uninstallFailed');
    btn.disabled = false;
  }
}

function memclean_init() {
  if (!_memclean_invoke) {
    _memclean_invoke = window.__TAURI__?.core?.invoke;
  }

  _memclean_build_region_ui(REGIONS);
  _memclean_load_mask();

  _mqs('memcleanCleanBtn')?.addEventListener('click', _memclean_do_clean);
  _mqs('memcleanSetupBtn')?.addEventListener('click', _memclean_do_setup);
  _mqs('memcleanUninstallBtn')?.addEventListener('click', _memclean_do_uninstall);

  memclean_refresh();
}


