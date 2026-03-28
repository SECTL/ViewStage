const DarkTheme = {
  name: 'dark',
  config: null,
  
  async load() {
    const response = await fetch('themes/dark/theme.json');
    this.config = await response.json();
    
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'themes/dark/theme.css';
    document.head.appendChild(link);
  },
  
  getIconPath(iconName) {
    const actualName = this.config?.icons?.[iconName] || iconName;
    return `themes/dark/icons/${actualName}.svg`;
  }
};

export default DarkTheme;
