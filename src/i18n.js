const i18n = {
    currentLocale: 'zh-CN',
    messages: {},
    
    async init() {
        const savedLocale = await this.getSavedLocale();
        if (savedLocale) {
            this.currentLocale = savedLocale;
        }
        await this.loadMessages(this.currentLocale);
        this.updatePageTexts();
        return this;
    },
    
    async getSavedLocale() {
        if (window.__TAURI__) {
            try {
                const { invoke } = window.__TAURI__.core;
                const settings = await invoke('get_settings');
                return settings.language || null;
            } catch (e) {
                console.error('Failed to get saved locale:', e);
            }
        }
        return localStorage.getItem('language') || null;
    },
    
    async loadMessages(locale) {
        try {
            const response = await fetch(`locales/${locale}.json`);
            if (!response.ok) {
                throw new Error(`Failed to load locale: ${locale}`);
            }
            this.messages = await response.json();
            this.currentLocale = locale;
        } catch (e) {
            console.error('Failed to load messages:', e);
            if (locale !== 'zh-CN') {
                await this.loadMessages('zh-CN');
            }
        }
    },
    
    t(key, params = {}) {
        const keys = key.split('.');
        let value = this.messages;
        
        for (const k of keys) {
            if (value && typeof value === 'object' && k in value) {
                value = value[k];
            } else {
                console.warn(`Translation not found: ${key}`);
                return key;
            }
        }
        
        if (typeof value !== 'string') {
            return key;
        }
        
        return value.replace(/\{(\w+)\}/g, (match, paramKey) => {
            return params[paramKey] !== undefined ? params[paramKey] : match;
        });
    },
    
    async setLocale(locale) {
        await this.loadMessages(locale);
        this.updatePageTexts();
        
        if (window.__TAURI__) {
            try {
                const { invoke } = window.__TAURI__.core;
                await invoke('save_settings', { settings: { language: locale } });
            } catch (e) {
                console.error('Failed to save locale:', e);
            }
        }
        localStorage.setItem('language', locale);
        
        document.documentElement.lang = locale;
    },
    
    updatePageTexts() {
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            el.textContent = this.t(key);
        });
        
        document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
            const key = el.getAttribute('data-i18n-placeholder');
            el.placeholder = this.t(key);
        });
        
        document.querySelectorAll('[data-i18n-title]').forEach(el => {
            const key = el.getAttribute('data-i18n-title');
            el.title = this.t(key);
        });
        
        document.querySelectorAll('[data-i18n-aria-label]').forEach(el => {
            const key = el.getAttribute('data-i18n-aria-label');
            el.setAttribute('aria-label', this.t(key));
        });
    },
    
    getLocale() {
        return this.currentLocale;
    },
    
    getSupportedLocales() {
        return [
            { code: 'zh-CN', name: '简体中文' },
            { code: 'zh-TW', name: '繁體中文' },
            { code: 'en-US', name: 'English' }
        ];
    }
};

window.i18n = i18n;
