/**
 * PyRust-ML Professional Dashboard JavaScript
 * Advanced interactivity and animations for professional ML dashboard
 */

// Performance monitoring and real-time updates
class DashboardController {
    constructor() {
        this.isInitialized = false;
        this.performanceMetrics = new Map();
        this.chartInstances = new Map();
        this.realTimeUpdates = true;
        
        this.init();
    }
    
    init() {
        if (this.isInitialized) return;
        
        this.setupEventListeners();
        this.initializeAnimations();
        this.startPerformanceMonitoring();
        this.setupThemeToggle();
        this.initializeTooltips();
        
        this.isInitialized = true;
        console.log('🚀 PyRust-ML Dashboard initialized');
    }
    
    setupEventListeners() {
        // Performance card hover effects
        document.addEventListener('DOMContentLoaded', () => {
            this.enhanceCards();
            this.setupRealTimeCounters();
            this.initializeProgressBars();
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case '1': this.switchTab(0); break;
                    case '2': this.switchTab(1); break;
                    case '3': this.switchTab(2); break;
                    case '4': this.switchTab(3); break;
                    case '5': this.switchTab(4); break;
                    case 'r': this.refreshDashboard(); break;
                }
            }
        });
    }
    
    enhanceCards() {
        const cards = document.querySelectorAll('.performance-card, .metric-card');
        
        cards.forEach((card, index) => {
            // Add entrance animation with delay
            card.style.animationDelay = `${index * 100}ms`;
            card.classList.add('animate-fadeInUp');
            
            // Enhanced hover effects
            card.addEventListener('mouseenter', () => {
                this.createRippleEffect(card, 'enter');
            });
            
            card.addEventListener('mouseleave', () => {
                this.createRippleEffect(card, 'leave');
            });
            
            // Click effects
            card.addEventListener('click', () => {
                this.createClickEffect(card);
            });
        });
    }
    
    createRippleEffect(element, action) {
        const ripple = document.createElement('div');
        ripple.className = `ripple-effect ${action}`;
        
        const rect = element.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        
        ripple.style.width = ripple.style.height = `${size}px`;
        ripple.style.left = `${-size / 2}px`;
        ripple.style.top = `${-size / 2}px`;
        
        element.style.position = 'relative';
        element.style.overflow = 'hidden';
        element.appendChild(ripple);
        
        setTimeout(() => {
            if (ripple.parentNode) {
                ripple.parentNode.removeChild(ripple);
            }
        }, 600);
    }
    
    createClickEffect(element) {
        element.style.transform = 'scale(0.98)';
        element.style.transition = 'transform 0.1s ease';
        
        setTimeout(() => {
            element.style.transform = '';
            element.style.transition = '';
        }, 100);
    }
    
    setupRealTimeCounters() {
        const counters = document.querySelectorAll('[data-counter]');
        
        counters.forEach(counter => {
            const target = parseInt(counter.getAttribute('data-counter'));
            const duration = 2000; // 2 seconds
            const increment = target / (duration / 16); // 60fps
            let current = 0;
            
            const updateCounter = () => {
                if (current < target) {
                    current += increment;
                    counter.textContent = Math.round(current);
                    requestAnimationFrame(updateCounter);
                } else {
                    counter.textContent = target;
                }
            };
            
            // Start counter when element comes into view
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        updateCounter();
                        observer.unobserve(entry.target);
                    }
                });
            });
            
            observer.observe(counter);
        });
    }
    
    initializeProgressBars() {
        const progressBars = document.querySelectorAll('.progress-bar');
        
        progressBars.forEach(bar => {
            const target = parseInt(bar.getAttribute('data-progress') || '0');
            
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        setTimeout(() => {
                            bar.style.width = `${target}%`;
                        }, 100);
                        observer.unobserve(entry.target);
                    }
                });
            });
            
            observer.observe(bar);
        });
    }
    
    startPerformanceMonitoring() {
        this.updatePerformanceMetrics();
        
        if (this.realTimeUpdates) {
            setInterval(() => {
                this.updatePerformanceMetrics();
            }, 5000); // Update every 5 seconds
        }
    }
    
    updatePerformanceMetrics() {
        // Simulate real-time performance data
        const metrics = {
            cpuUsage: Math.random() * 30 + 40, // 40-70%
            memoryUsage: Math.random() * 20 + 50, // 50-70%
            gpuUsage: Math.random() * 40 + 10, // 10-50%
            temperature: Math.random() * 15 + 55, // 55-70°C
            powerUsage: Math.random() * 50 + 100, // 100-150W
        };
        
        this.performanceMetrics.set('current', metrics);
        this.updateMetricDisplays(metrics);
    }
    
    updateMetricDisplays(metrics) {
        Object.entries(metrics).forEach(([key, value]) => {
            const elements = document.querySelectorAll(`[data-metric="${key}"]`);
            elements.forEach(element => {
                if (element.tagName === 'PROGRESS') {
                    element.value = value;
                } else {
                    element.textContent = `${Math.round(value)}${this.getMetricUnit(key)}`;
                }
                
                // Add color coding
                this.updateMetricColor(element, value, key);
            });
        });
    }
    
    getMetricUnit(metric) {
        const units = {
            cpuUsage: '%',
            memoryUsage: '%',
            gpuUsage: '%',
            temperature: '°C',
            powerUsage: 'W'
        };
        return units[metric] || '';
    }
    
    updateMetricColor(element, value, metric) {
        const thresholds = {
            cpuUsage: { warning: 70, danger: 85 },
            memoryUsage: { warning: 70, danger: 85 },
            gpuUsage: { warning: 80, danger: 95 },
            temperature: { warning: 70, danger: 80 },
            powerUsage: { warning: 140, danger: 160 }
        };
        
        const threshold = thresholds[metric];
        if (!threshold) return;
        
        element.classList.remove('text-green-500', 'text-yellow-500', 'text-red-500');
        
        if (value >= threshold.danger) {
            element.classList.add('text-red-500');
        } else if (value >= threshold.warning) {
            element.classList.add('text-yellow-500');
        } else {
            element.classList.add('text-green-500');
        }
    }
    
    setupThemeToggle() {
        const themeToggle = document.createElement('button');
        themeToggle.className = 'theme-toggle';
        themeToggle.innerHTML = '🌙';
        themeToggle.setAttribute('aria-label', 'Toggle dark mode');
        
        themeToggle.addEventListener('click', () => {
            this.toggleTheme();
        });
        
        // Add to header if it exists
        const header = document.querySelector('.dashboard-header');
        if (header) {
            header.appendChild(themeToggle);
        }
    }
    
    toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        
        // Update theme toggle icon
        const themeToggle = document.querySelector('.theme-toggle');
        if (themeToggle) {
            themeToggle.innerHTML = newTheme === 'dark' ? '☀️' : '🌙';
        }
        
        // Animate theme transition
        document.body.style.transition = 'background-color 0.3s ease, color 0.3s ease';
        setTimeout(() => {
            document.body.style.transition = '';
        }, 300);
    }
    
    initializeTooltips() {
        const tooltipElements = document.querySelectorAll('[data-tooltip]');
        
        tooltipElements.forEach(element => {
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = element.getAttribute('data-tooltip');
            document.body.appendChild(tooltip);
            
            element.addEventListener('mouseenter', (e) => {
                this.showTooltip(tooltip, e);
            });
            
            element.addEventListener('mouseleave', () => {
                this.hideTooltip(tooltip);
            });
            
            element.addEventListener('mousemove', (e) => {
                this.updateTooltipPosition(tooltip, e);
            });
        });
    }
    
    showTooltip(tooltip, event) {
        tooltip.style.opacity = '1';
        tooltip.style.visibility = 'visible';
        this.updateTooltipPosition(tooltip, event);
    }
    
    hideTooltip(tooltip) {
        tooltip.style.opacity = '0';
        tooltip.style.visibility = 'hidden';
    }
    
    updateTooltipPosition(tooltip, event) {
        const x = event.clientX + 10;
        const y = event.clientY - 30;
        
        tooltip.style.left = `${x}px`;
        tooltip.style.top = `${y}px`;
    }
    
    switchTab(index) {
        const tabs = document.querySelectorAll('[data-baseweb="tab"]');
        if (tabs[index]) {
            tabs[index].click();
        }
    }
    
    refreshDashboard() {
        // Add refresh animation
        const refreshIcon = document.createElement('div');
        refreshIcon.innerHTML = '🔄';
        refreshIcon.className = 'refresh-indicator';
        document.body.appendChild(refreshIcon);
        
        setTimeout(() => {
            refreshIcon.remove();
            location.reload();
        }, 1000);
    }
    
    // Chart enhancement methods
    enhanceCharts() {
        // This will be called by Plotly charts when they're created
        const charts = document.querySelectorAll('.js-plotly-plot');
        
        charts.forEach(chart => {
            this.addChartInteractivity(chart);
        });
    }
    
    addChartInteractivity(chart) {
        // Add hover effects to chart containers
        chart.addEventListener('mouseenter', () => {
            chart.style.transform = 'scale(1.02)';
            chart.style.transition = 'transform 0.3s ease';
        });
        
        chart.addEventListener('mouseleave', () => {
            chart.style.transform = 'scale(1)';
        });
    }
    
    // Notification system
    showNotification(message, type = 'info', duration = 3000) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-icon">${this.getNotificationIcon(type)}</span>
                <span class="notification-message">${message}</span>
                <button class="notification-close">&times;</button>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.classList.add('notification-show');
        }, 10);
        
        // Auto remove
        setTimeout(() => {
            this.hideNotification(notification);
        }, duration);
        
        // Close button
        notification.querySelector('.notification-close').addEventListener('click', () => {
            this.hideNotification(notification);
        });
    }
    
    hideNotification(notification) {
        notification.classList.remove('notification-show');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }
    
    getNotificationIcon(type) {
        const icons = {
            info: 'ℹ️',
            success: '✅',
            warning: '⚠️',
            error: '❌'
        };
        return icons[type] || icons.info;
    }
    
    // Utility methods for Streamlit integration
    onStreamlitUpdate() {
        // Re-initialize when Streamlit updates
        setTimeout(() => {
            this.enhanceCards();
            this.setupRealTimeCounters();
            this.initializeProgressBars();
            this.enhanceCharts();
        }, 100);
    }
}

// Advanced CSS injection for additional styles
function injectAdvancedStyles() {
    const styles = `
        .ripple-effect {
            position: absolute;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            pointer-events: none;
            transform: scale(0);
            animation: ripple 0.6s linear;
        }
        
        @keyframes ripple {
            to {
                transform: scale(4);
                opacity: 0;
            }
        }
        
        .theme-toggle {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            width: 50px;
            height: 50px;
            font-size: 20px;
            cursor: pointer;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
            z-index: 1000;
        }
        
        .theme-toggle:hover {
            background: rgba(255, 255, 255, 0.3);
            transform: scale(1.1);
        }
        
        .tooltip {
            position: fixed;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 14px;
            pointer-events: none;
            z-index: 1000;
            opacity: 0;
            visibility: hidden;
            transition: all 0.2s ease;
        }
        
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            border-left: 4px solid #3b82f6;
            min-width: 300px;
            max-width: 400px;
            z-index: 1000;
            transform: translateX(100%);
            transition: transform 0.3s ease;
        }
        
        .notification-show {
            transform: translateX(0);
        }
        
        .notification-success {
            border-left-color: #10b981;
        }
        
        .notification-warning {
            border-left-color: #f59e0b;
        }
        
        .notification-error {
            border-left-color: #ef4444;
        }
        
        .notification-content {
            display: flex;
            align-items: center;
            padding: 16px;
            gap: 12px;
        }
        
        .notification-icon {
            font-size: 20px;
            flex-shrink: 0;
        }
        
        .notification-message {
            flex: 1;
            font-weight: 500;
        }
        
        .notification-close {
            background: none;
            border: none;
            font-size: 20px;
            cursor: pointer;
            opacity: 0.5;
            transition: opacity 0.2s ease;
        }
        
        .notification-close:hover {
            opacity: 1;
        }
        
        .refresh-indicator {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 48px;
            animation: spin 1s linear infinite;
            z-index: 1000;
        }
        
        /* Dark theme styles */
        [data-theme="dark"] {
            --glass-bg: rgba(0, 0, 0, 0.3);
            --glass-border: rgba(255, 255, 255, 0.1);
        }
        
        [data-theme="dark"] .notification {
            background: #1f2937;
            color: white;
        }
        
        [data-theme="dark"] .tooltip {
            background: rgba(255, 255, 255, 0.9);
            color: #1f2937;
        }
        
        /* Performance indicators */
        .performance-indicator {
            position: relative;
            overflow: hidden;
        }
        
        .performance-indicator::after {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
    `;
    
    const styleSheet = document.createElement('style');
    styleSheet.textContent = styles;
    document.head.appendChild(styleSheet);
}

// Initialize dashboard when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        injectAdvancedStyles();
        window.dashboardController = new DashboardController();
    });
} else {
    injectAdvancedStyles();
    window.dashboardController = new DashboardController();
}

// Streamlit integration helpers
window.streamlitReady = function() {
    if (window.dashboardController) {
        window.dashboardController.onStreamlitUpdate();
    }
};

// Export for external use
window.PyRustMLDashboard = {
    controller: () => window.dashboardController,
    showNotification: (message, type, duration) => {
        if (window.dashboardController) {
            window.dashboardController.showNotification(message, type, duration);
        }
    },
    refreshMetrics: () => {
        if (window.dashboardController) {
            window.dashboardController.updatePerformanceMetrics();
        }
    }
};

console.log('🚀 PyRust-ML Professional Dashboard JavaScript loaded');