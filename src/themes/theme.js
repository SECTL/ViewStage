const ThemeManager = {
  currentTheme: null,
  themeLink: null,

  init(themeName = 'dark') {
    this.themeLink = document.getElementById('theme-style');
    if (!this.themeLink) {
      this.themeLink = document.createElement('link');
      this.themeLink.id = 'theme-style';
      this.themeLink.rel = 'stylesheet';
      document.head.appendChild(this.themeLink);
    }
    this.setTheme(themeName);
  },

  setTheme(themeName) {
    const themePath = `themes/${themeName}.css`;
    this.themeLink.href = themePath;
    this.currentTheme = themeName;
  },

  getTheme() {
    return this.currentTheme;
  }
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => ThemeManager.init());
} else {
  ThemeManager.init();
}

export default ThemeManager;
