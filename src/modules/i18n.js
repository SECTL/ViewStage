/**
 * 国际化模块 - 提供多语言翻译、语言切换、页面文本渲染
 */
const i18n = {
    current_locale: 'zh-CN',
    messages: {},

    /**
     * 初始化国际化：读取已保存语言 → 加载翻译文件 → 渲染页面文本
     * @returns {Promise<Object>} i18n 实例
     */
    async init_start() {
        const saved_locale = await this.fetch_saved_locale();
        if (saved_locale) {
            this.current_locale = saved_locale;
        }
        await this.load_messages(this.current_locale);
        this.render_page_texts();
        return this;
    },

    // 从配置或 localStorage 读取已保存的语言设置
    async fetch_saved_locale() {
        if (window.__TAURI__) {
            try {
                const { invoke } = window.__TAURI__.core;
                const result = await window.getSettings();
                return result.settings.language || null;
            } catch (e) {
                console.error('Failed to get saved locale:', e);
            }
        }
        return localStorage.getItem('language') || null;
    },

    // 加载指定语言的翻译 JSON 文件，失败时回退到 zh-CN
    async load_messages(locale) {
        try {
            const response = await fetch(`locales/${locale}.json`);
            if (!response.ok) {
                throw new Error(`Failed to load locale: ${locale}`);
            }
            this.messages = await response.json();
            this.current_locale = locale;
        } catch (e) {
            console.error('Failed to load messages:', e);
            if (locale !== 'zh-CN') {
                await this.load_messages('zh-CN');
            }
        }
    },

    /**
     * 根据 key 获取翻译文本，支持 {param} 模板替换
     * @param {string} key - 翻译键路径（如 'errors.initFailed'）
     * @param {Object} [params] - 模板替换参数
     * @returns {string} 翻译后的字符串，未找到时返回 key 本身
     */
    format_translate(key, params = {}) {
        const keys_list = key.split('.');
        let value = this.messages;

        for (const key_item of keys_list) {
            if (value && typeof value === 'object' && key_item in value) {
                value = value[key_item];
            } else {
                console.warn(`Translation not found: ${key}`);
                return key;
            }
        }

        if (typeof value !== 'string') {
            return key;
        }

        return value.replace(/\{(\w+)\}/g, (match, param_key) => {
            return params[param_key] !== undefined ? params[param_key] : match;
        });
    },

    /**
     * 切换语言：加载翻译 → 渲染页面 → 持久化保存到配置和 localStorage
     * @param {string} locale - 语言代码（如 'zh-CN'）
     */
    async update_locale(locale) {
        await this.load_messages(locale);
        this.render_page_texts();

        if (window.__TAURI__) {
            try {
                const { invoke } = window.__TAURI__.core;
                await invoke('settings_save_all', { settings: { language: locale } });
            } catch (e) {
                console.error('Failed to save locale:', e);
            }
        }
        localStorage.setItem('language', locale);

        document.documentElement.lang = locale;
    },

    // 遍历 DOM，替换 [data-i18n] 等属性对应的文本内容
    render_page_texts() {
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            el.textContent = this.format_translate(key);
        });

        document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
            const key = el.getAttribute('data-i18n-placeholder');
            el.placeholder = this.format_translate(key);
        });

        document.querySelectorAll('[data-i18n-title]').forEach(el => {
            const key = el.getAttribute('data-i18n-title');
            el.title = this.format_translate(key);
        });

        document.querySelectorAll('[data-i18n-aria-label]').forEach(el => {
            const key = el.getAttribute('data-i18n-aria-label');
            el.setAttribute('aria-label', this.format_translate(key));
        });
    },

};

window.i18n = i18n;
