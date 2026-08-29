/*
 * settings_cache.js — settings_fetch_all 结果缓存
 *
 * 作为普通 <script> 引入各窗口（index.html / settings.html / splashscreen.html），
 * 把 getSettings / invalidateSettingsCache 挂到 window，供业务模块直接使用，
 * 避免多个模块、多个窗口各自重复 IPC 读盘 config.json。
 *
 * - 并发调用：首个调用发起 IPC，其余共享同一个 in-flight Promise（启动期去重）；
 * - 后续调用：直接返回已缓存的结果（0 次 IPC）；
 * - 设置保存后：监听 settings-changed 自动失效，下次读取拿到最新值。
 *
 * 注意：ES 模块在「每个窗口上下文」内是单例，因此缓存作用域为单个窗口，
 * 这正好符合预期（每个窗口独立读一次配置并去重）。
 */
(function () {
    'use strict';

    var _cache = null;
    var _inflight = null;

    function _rawFetch() {
        var core = window.__TAURI__ && window.__TAURI__.core;
        var invoke = core && core.invoke;
        if (typeof invoke !== 'function') return Promise.resolve(null);
        return invoke('settings_fetch_all');
    }

    function getSettings(force) {
        if (force) {
            _cache = null;
            _inflight = null;
        }
        if (_cache) return Promise.resolve(_cache);
        if (_inflight) return _inflight;
        _inflight = _rawFetch().then(function (s) {
            _cache = s;
            _inflight = null;
            return s;
        }).catch(function (err) {
            _inflight = null;
            throw err;
        });
        return _inflight;
    }

    function invalidateSettingsCache() {
        _cache = null;
        _inflight = null;
    }

    window.getSettings = getSettings;
    window.invalidateSettingsCache = invalidateSettingsCache;

    // 设置保存后使缓存失效，保证下次读取为最新值（不影响并发去重）
    try {
        var ev = window.__TAURI__ && window.__TAURI__.event;
        if (ev && typeof ev.listen === 'function') {
            ev.listen('settings-changed', function () {
                invalidateSettingsCache();
            });
        }
    } catch (e) {
        /* 忽略：非 Tauri 环境下无需监听 */
    }
})();
