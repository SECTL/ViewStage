const SimplifyTheme = {
  name: 'simplify',
  config: null,
  
  getBasePath() {
    const parts = window.location.pathname.split('/').filter(p => p);
    const depth = Math.max(0, parts.length - 1);
    return '../'.repeat(depth);
  },
  
  async load() {
    const base = this.getBasePath();
    const response = await fetch(`${base}themes/simplify/theme.json`);
    this.config = await response.json();
    
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = `${base}themes/simplify/theme.css`;
    document.head.appendChild(link);
  },
  
  getIconPath(iconName) {
    const actualName = this.config?.icons?.[iconName] || iconName;
    const base = this.getBasePath();
    return `${base}themes/simplify/icons/${actualName}.svg`;
  },
  
  getShowToolbarText() {
    return this.config?.showToolbarText !== false;
  },
  
  getCanvasBgColor() {
    return this.config?.canvasBgColor || '#ffffff';
  }
};

export default SimplifyTheme;
