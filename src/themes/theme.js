const ThemeManager = {
  currentTheme: null,
  currentThemeModule: null,
  userThemePath: null,
  loadCss: true,

  async init(themeName = 'dark') {
    this.loadCss = !window.location.pathname.includes('settings.html');
    await this.setTheme(themeName);
  },

  async setTheme(themeName) {
    try {
      let themeModule = null;
      
      if (window.__TAURI__ && !this.isBuiltInTheme(themeName)) {
        if (!this.userThemePath) {
          const { invoke } = window.__TAURI__.core;
          try {
            this.userThemePath = await invoke('get_theme_dir');
          } catch (e) {
            console.warn('无法获取用户主题目录:', e);
          }
        }
        
        if (this.userThemePath) {
          const userThemeDir = `${this.userThemePath}/${themeName}`;
          const hasUserTheme = await this.checkUserTheme(userThemeDir);
          
          if (hasUserTheme) {
            themeModule = await this.loadUserTheme(userThemeDir, themeName);
          }
        }
      }
      
      if (!themeModule) {
        const module = await import(`./${themeName}/theme.js`);
        themeModule = module.default;
      }
      
      this.currentThemeModule = themeModule;
      this.currentTheme = themeName;
      
      if (this.loadCss && this.currentThemeModule.load) {
        await this.currentThemeModule.load();
      }
      this.loadIcons();
    } catch (error) {
      console.error(`Failed to load theme: ${themeName}`, error);
    }
  },

  isBuiltInTheme(themeName) {
    const builtInThemes = ['dark'];
    return builtInThemes.includes(themeName);
  },

  async checkUserTheme(themeDir) {
    if (!window.__TAURI__) return false;
    
    const { fs } = window.__TAURI__;
    try {
      const configPath = `${themeDir}/theme.json`;
      const content = await fs.readTextFile(configPath);
      return !!content;
    } catch {
      return false;
    }
  },

  async loadUserTheme(themeDir, themeName) {
    const { fs, convertFileSrc } = window.__TAURI__;
    
    const configPath = `${themeDir}/theme.json`;
    const configContent = await fs.readTextFile(configPath);
    const config = JSON.parse(configContent);
    
    return {
      name: themeName,
      config: config,
      themeDir: themeDir,
      
      async load() {
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = convertFileSrc(`${this.themeDir}/theme.css`);
        document.head.appendChild(link);
      },
      
      getIconPath(iconName) {
        const actualName = this.config?.icons?.[iconName] || iconName;
        return convertFileSrc(`${this.themeDir}/icons/${actualName}.svg`);
      }
    };
  },

  getTheme() {
    return this.currentTheme;
  },

  getIconPath(iconName) {
    if (this.currentThemeModule && this.currentThemeModule.getIconPath) {
      return this.currentThemeModule.getIconPath(iconName);
    }
    return `themes/${this.currentTheme}/icons/${iconName}.svg`;
  },

  getIcon(iconName, options = {}) {
    const { width = 16, height = 16, alt = '', style = '' } = options;
    const src = this.getIconPath(iconName);
    return `<img src="${src}" width="${width}" height="${height}" alt="${alt}" style="${style}">`;
  },

  loadIcons() {
    const icons = document.querySelectorAll('[data-icon]');
    icons.forEach(img => {
      const iconName = img.getAttribute('data-icon');
      img.src = this.getIconPath(iconName);
    });
  }
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => ThemeManager.init());
} else {
  ThemeManager.init();
}

export default ThemeManager;
