const SimplifyTheme = {
  name: 'simplify',
  config: null,
  
  async load() {
    const response = await fetch('themes/simplify/theme.json');
    this.config = await response.json();
    
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'themes/simplify/theme.css';
    document.head.appendChild(link);
  },
  
  getIconPath(iconName) {
    const actualName = this.config?.icons?.[iconName] || iconName;
    return `themes/simplify/icons/${actualName}.svg`;
  },
  
  getShowToolbarText() {
    return this.config?.showToolbarText !== false;
  },
  
  getCanvasBgColor() {
    return this.config?.canvasBgColor || '#ffffff';
  }
};

export default SimplifyTheme;
