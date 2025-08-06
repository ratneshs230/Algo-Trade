/**
 * Dashboard JavaScript - Trading Interface Frontend
 * 
 * Handles all frontend interactions including Kite login,
 * status updates, and navigation between sections.
 */

class TradingDashboard {
    constructor() {
        this.currentSection = 'kite-login';
        this.loginStatusInterval = null;
        this.connectionStatusInterval = null;
        this.ratingsRefreshInterval = null;
        
        this.init();
    }

    /**
     * Initialize the dashboard
     */
    init() {
        this.setupEventListeners();
        this.setupNavigation();
        this.checkConnectionStatus();
        this.loadInitialData();
        
        
        console.log('Trading Dashboard initialized');
    }
    

    /**
     * Setup event listeners for all interactive elements
     */
    setupEventListeners() {
        // Start login button
        const startLoginBtn = document.getElementById('start-login-btn');
        if (startLoginBtn) {
            startLoginBtn.addEventListener('click', () => this.startKiteLogin());
        }

        // Submit TOTP button
        const submitTotpBtn = document.getElementById('submit-totp-btn');
        if (submitTotpBtn) {
            submitTotpBtn.addEventListener('click', () => this.submitTotp());
        }

        // TOTP input field (submit on Enter)
        const totpInput = document.getElementById('totp-input');
        if (totpInput) {
            totpInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.submitTotp();
                }
            });

            // Auto-format TOTP input (only digits)
            totpInput.addEventListener('input', (e) => {
                e.target.value = e.target.value.replace(/\D/g, '').substring(0, 6);
            });
        }

        // Refresh ratings button
        const refreshRatingsBtn = document.getElementById('refresh-ratings-btn');
        if (refreshRatingsBtn) {
            refreshRatingsBtn.addEventListener('click', () => this.refreshRatings());
        }

        // Stock detail modal close button
        const closeDetailBtn = document.getElementById('close-detail-btn');
        if (closeDetailBtn) {
            closeDetailBtn.addEventListener('click', () => this.closeStockDetail());
        }

        // Click outside modal to close
        const stockDetailOverlay = document.getElementById('stock-detail-overlay');
        if (stockDetailOverlay) {
            stockDetailOverlay.addEventListener('click', (e) => {
                if (e.target === stockDetailOverlay) {
                    this.closeStockDetail();
                }
            });
        }

        // Tab switching in stock detail modal
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('tab-btn')) {
                this.switchDetailTab(e.target.getAttribute('data-tab'));
            }
        });
    }

    /**
     * Setup navigation between different sections
     */
    setupNavigation() {
        const navLinks = document.querySelectorAll('.nav-link');
        
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                
                const sectionId = link.getAttribute('data-section');
                this.switchSection(sectionId);
                
                // Update active navigation item
                document.querySelectorAll('.nav-item').forEach(item => {
                    item.classList.remove('active');
                });
                link.closest('.nav-item').classList.add('active');
            });
        });
    }

    /**
     * Switch between different dashboard sections
     */
    switchSection(sectionId) {
        // Hide all sections
        document.querySelectorAll('.content-section').forEach(section => {
            section.classList.remove('active');
        });

        // Show target section
        const targetSection = document.getElementById(sectionId);
        if (targetSection) {
            targetSection.classList.add('active');
            this.currentSection = sectionId;
        }
    }

    /**
     * Check and update connection status
     */
    checkConnectionStatus() {
        const statusIcon = document.getElementById('connection-status');
        const statusText = document.getElementById('connection-text');

        // Simple connection check - you can enhance this
        const updateStatus = (isOnline) => {
            if (isOnline) {
                statusIcon.classList.remove('offline');
                statusText.textContent = 'Connected';
            } else {
                statusIcon.classList.add('offline');
                statusText.textContent = 'Disconnected';
            }
        };

        // Initial check
        updateStatus(navigator.onLine);

        // Listen for connection changes
        window.addEventListener('online', () => updateStatus(true));
        window.addEventListener('offline', () => updateStatus(false));

        // Periodic server connection check
        this.connectionStatusInterval = setInterval(() => {
            fetch('/api/config/check')
                .then(response => response.json())
                .then(data => {
                    updateStatus(data.success);
                })
                .catch(() => {
                    updateStatus(false);
                });
        }, 30000); // Check every 30 seconds
    }

    /**
     * Load initial data when dashboard starts
     */
    async loadInitialData() {
        await this.loadConfigStatus();
        await this.loadTokenInfo();
    }

    /**
     * Load and display configuration status
     */
    async loadConfigStatus() {
        const configStatusEl = document.getElementById('config-status');
        
        try {
            const response = await fetch('/api/config/check');
            const data = await response.json();
            
            if (data.success) {
                const configItems = Object.entries(data.config_status).map(([key, configured]) => {
                    const statusClass = configured ? 'configured' : 'not-configured';
                    const statusIcon = configured ? 'fas fa-check-circle' : 'fas fa-times-circle';
                    const statusText = configured ? 'Configured' : 'Not Configured';
                    
                    return `
                        <div class="config-item">
                            <span class="config-label">${key.replace('_', ' ')}</span>
                            <span class="config-status ${statusClass}">
                                <i class="${statusIcon}"></i>
                                ${statusText}
                            </span>
                        </div>
                    `;
                }).join('');

                const overallStatus = data.configured ? 'All Required' : 'Missing Required';
                const overallClass = data.configured ? 'text-success' : 'text-error';

                configStatusEl.innerHTML = `
                    <div class="config-item">
                        <span class="config-label"><strong>Overall Status</strong></span>
                        <span class="config-status ${overallClass}">
                            <strong>${overallStatus}</strong>
                        </span>
                    </div>
                    ${configItems}
                `;
            } else {
                configStatusEl.innerHTML = `
                    <div class="status-message error">
                        <i class="fas fa-exclamation-triangle"></i>
                        <span>Error checking configuration: ${data.error}</span>
                    </div>
                `;
            }
        } catch (error) {
            configStatusEl.innerHTML = `
                <div class="status-message error">
                    <i class="fas fa-exclamation-triangle"></i>
                    <span>Failed to load configuration status</span>
                </div>
            `;
        }
    }

    /**
     * Load and display current token information
     */
    async loadTokenInfo() {
        const tokenInfoEl = document.getElementById('token-info');
        
        try {
            const response = await fetch('/api/kite/token_info');
            const data = await response.json();
            
            if (data.success && data.token_info) {
                const tokenInfo = data.token_info;
                tokenInfoEl.innerHTML = `
                    <div class="token-item">
                        <span class="token-label">Access Token</span>
                        <span class="token-value">${tokenInfo.access_token}</span>
                    </div>
                    <div class="token-item">
                        <span class="token-label">Date Generated</span>
                        <span class="token-value">${tokenInfo.date}</span>
                    </div>
                    <div class="token-item">
                        <span class="token-label">Timestamp</span>
                        <span class="token-value">${tokenInfo.timestamp}</span>
                    </div>
                `;
            } else {
                tokenInfoEl.innerHTML = `
                    <div class="status-message warning">
                        <i class="fas fa-info-circle"></i>
                        <span>${data.message || 'No access token found'}</span>
                    </div>
                `;
            }
        } catch (error) {
            tokenInfoEl.innerHTML = `
                <div class="status-message error">
                    <i class="fas fa-exclamation-triangle"></i>
                    <span>Failed to load token information</span>
                </div>
            `;
        }
    }

    /**
     * Start the Kite login process
     */
    async startKiteLogin() {
        const startBtn = document.getElementById('start-login-btn');
        const startSection = document.getElementById('login-start-section');
        const statusEl = document.getElementById('login-status');
        const statusText = document.getElementById('status-text');
        const progressBar = document.getElementById('status-progress');

        try {
            // Disable start button
            startBtn.disabled = true;
            startBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';

            // Show progress
            progressBar.classList.remove('hidden');
            this.updateStatus('Starting Kite login process...', 'info');

            const response = await fetch('/api/kite/start_login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const data = await response.json();

            if (data.success) {
                this.updateStatus('Login process started successfully', 'success');
                
                // Start polling for status updates
                this.startStatusPolling();
            } else {
                throw new Error(data.error || 'Failed to start login process');
            }

        } catch (error) {
            this.updateStatus(`Error: ${error.message}`, 'error');
            progressBar.classList.add('hidden');
            
            // Re-enable start button
            startBtn.disabled = false;
            startBtn.innerHTML = '<i class="fas fa-play"></i> Start Kite Login Process';
        }
    }

    /**
     * Submit TOTP code
     */
    async submitTotp() {
        const totpInput = document.getElementById('totp-input');
        const submitBtn = document.getElementById('submit-totp-btn');
        const totpCode = totpInput.value.trim();

        // Validate TOTP code
        if (!totpCode || totpCode.length !== 6 || !/^\d{6}$/.test(totpCode)) {
            this.updateStatus('Please enter a valid 6-digit TOTP code', 'error');
            totpInput.focus();
            return;
        }

        try {
            // Disable submit button
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Submitting...';

            this.updateStatus('Processing TOTP code...', 'info');

            const response = await fetch('/api/kite/submit_totp', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    totp_code: totpCode
                })
            });

            const data = await response.json();

            if (data.success) {
                this.updateStatus('TOTP submitted successfully. Processing...', 'success');
                totpInput.value = '';
                
                // Continue polling for final result
            } else {
                throw new Error(data.error || 'Failed to submit TOTP');
            }

        } catch (error) {
            this.updateStatus(`Error: ${error.message}`, 'error');
            
            // Re-enable submit button
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<i class="fas fa-check"></i> Submit';
            totpInput.focus();
        }
    }

    /**
     * Start polling for login status updates
     */
    startStatusPolling() {
        // Clear any existing interval
        if (this.loginStatusInterval) {
            clearInterval(this.loginStatusInterval);
        }

        this.loginStatusInterval = setInterval(async () => {
            try {
                const response = await fetch('/api/kite/status');
                const status = await response.json();

                this.handleStatusUpdate(status);

            } catch (error) {
                console.error('Error polling status:', error);
            }
        }, 2000); // Poll every 2 seconds
    }

    /**
     * Handle status updates from the server
     */
    handleStatusUpdate(status) {
        const { status: currentStatus, message, error } = status;

        switch (currentStatus) {
            case 'idle':
                this.updateStatus('Ready to start login process', 'info');
                break;

            case 'starting':
                this.updateStatus('Initializing login process...', 'info');
                break;

            case 'waiting_for_totp':
                this.updateStatus('Please enter your TOTP code below', 'warning');
                this.showTotpSection();
                break;

            case 'processing':
                this.updateStatus('Processing authentication...', 'info');
                break;

            case 'success':
                this.updateStatus('Access token generated successfully!', 'success');
                this.handleLoginSuccess(status);
                break;

            case 'error':
                this.updateStatus(`Error: ${error || message}`, 'error');
                this.handleLoginError();
                break;

            default:
                if (message) {
                    this.updateStatus(message, 'info');
                }
        }
    }

    /**
     * Show TOTP input section
     */
    showTotpSection() {
        const startSection = document.getElementById('login-start-section');
        const totpSection = document.getElementById('totp-section');

        startSection.classList.add('hidden');
        totpSection.classList.remove('hidden');

        // Focus on TOTP input
        const totpInput = document.getElementById('totp-input');
        if (totpInput) {
            totpInput.focus();
        }
    }

    /**
     * Handle successful login
     */
    handleLoginSuccess(status) {
        // Stop polling
        if (this.loginStatusInterval) {
            clearInterval(this.loginStatusInterval);
            this.loginStatusInterval = null;
        }

        // Hide progress bar
        document.getElementById('status-progress').classList.add('hidden');

        // Reset sections
        setTimeout(() => {
            this.resetLoginSections();
            this.loadTokenInfo(); // Reload token info
        }, 3000);
    }

    /**
     * Handle login error
     */
    handleLoginError() {
        // Stop polling
        if (this.loginStatusInterval) {
            clearInterval(this.loginStatusInterval);
            this.loginStatusInterval = null;
        }

        // Hide progress bar
        document.getElementById('status-progress').classList.add('hidden');

        // Reset sections after delay
        setTimeout(() => {
            this.resetLoginSections();
        }, 5000);
    }

    /**
     * Reset login sections to initial state
     */
    resetLoginSections() {
        const startSection = document.getElementById('login-start-section');
        const totpSection = document.getElementById('totp-section');
        const startBtn = document.getElementById('start-login-btn');
        const submitBtn = document.getElementById('submit-totp-btn');
        const totpInput = document.getElementById('totp-input');

        // Show start section, hide TOTP section
        startSection.classList.remove('hidden');
        totpSection.classList.add('hidden');

        // Reset buttons
        startBtn.disabled = false;
        startBtn.innerHTML = '<i class="fas fa-play"></i> Start Kite Login Process';
        
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fas fa-check"></i> Submit';

        // Clear TOTP input
        if (totpInput) {
            totpInput.value = '';
        }

        // Reset status
        this.updateStatus('Ready to start login process', 'info');
    }

    /**
     * Update status message display
     */
    updateStatus(message, type = 'info') {
        const statusMessage = document.querySelector('#login-status .status-message');
        const statusText = document.getElementById('status-text');

        if (!statusMessage || !statusText) return;

        // Remove existing type classes
        statusMessage.classList.remove('success', 'error', 'warning', 'info');
        
        // Add new type class
        if (type !== 'info') {
            statusMessage.classList.add(type);
        }

        // Update icon based on type
        const icon = statusMessage.querySelector('i');
        if (icon) {
            icon.className = this.getStatusIcon(type);
        }

        // Update text
        statusText.textContent = message;

        console.log(`Status: ${type} - ${message}`);
    }

    /**
     * Get appropriate icon for status type
     */
    getStatusIcon(type) {
        const icons = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };
        return icons[type] || icons.info;
    }

    /**
     * Live Ratings Functions
     */

    /**
     * Switch section and load ratings if switching to live-ratings
     */
    switchSection(sectionId) {
        // Hide all sections
        document.querySelectorAll('.content-section').forEach(section => {
            section.classList.remove('active');
        });

        // Show target section
        const targetSection = document.getElementById(sectionId);
        if (targetSection) {
            targetSection.classList.add('active');
            this.currentSection = sectionId;

            // Load ratings when switching to live ratings section
            if (sectionId === 'live-ratings') {
                this.loadLiveRatings();
                this.startRatingsAutoRefresh();
            } else {
                this.stopRatingsAutoRefresh();
            }
        }
    }

    /**
     * Load and display live ratings
     */
    async loadLiveRatings() {
        try {
            await Promise.all([
                this.loadRatingsSummary(),
                this.loadTopBottomRatings()
            ]);
        } catch (error) {
            console.error('Error loading live ratings:', error);
        }
    }

    /**
     * Load ratings summary
     */
    async loadRatingsSummary() {
        const summaryEl = document.getElementById('ratings-summary');
        
        try {
            const response = await fetch('/api/ratings/summary');
            const data = await response.json();
            
            if (data.success) {
                const summary = data.data;
                const lastUpdate = new Date(summary.timestamp).toLocaleString();
                
                summaryEl.innerHTML = `
                    <div class="ratings-summary">
                        <div class="summary-stat">
                            <h4>${summary.total_stocks}</h4>
                            <p>Total Stocks</p>
                        </div>
                        <div class="summary-stat positive">
                            <h4>${summary.highest_score.toFixed(2)}</h4>
                            <p>Highest Score</p>
                        </div>
                        <div class="summary-stat negative">
                            <h4>${summary.lowest_score.toFixed(2)}</h4>
                            <p>Lowest Score</p>
                        </div>
                        <div class="summary-stat neutral">
                            <h4>${summary.average_score.toFixed(2)}</h4>
                            <p>Average Score</p>
                        </div>
                    </div>
                    
                    <div class="ratings-distribution">
                        <div class="distribution-item strong-buy">
                            <div class="distribution-count">${summary.rating_distribution.strong_buy}</div>
                            <div class="distribution-label">Strong Buy</div>
                        </div>
                        <div class="distribution-item buy">
                            <div class="distribution-count">${summary.rating_distribution.buy}</div>
                            <div class="distribution-label">Buy</div>
                        </div>
                        <div class="distribution-item neutral">
                            <div class="distribution-count">${summary.rating_distribution.neutral}</div>
                            <div class="distribution-label">Neutral</div>
                        </div>
                        <div class="distribution-item sell">
                            <div class="distribution-count">${summary.rating_distribution.sell}</div>
                            <div class="distribution-label">Sell</div>
                        </div>
                        <div class="distribution-item strong-sell">
                            <div class="distribution-count">${summary.rating_distribution.strong_sell}</div>
                            <div class="distribution-label">Strong Sell</div>
                        </div>
                    </div>
                    
                    <div class="last-updated">
                        <i class="fas fa-clock"></i>
                        Last updated: ${lastUpdate}
                    </div>
                `;
            } else {
                summaryEl.innerHTML = `
                    <div class="ratings-error">
                        <i class="fas fa-exclamation-triangle"></i>
                        <h3>Error Loading Summary</h3>
                        <p>${data.message || 'Unable to load ratings summary'}</p>
                    </div>
                `;
            }
        } catch (error) {
            summaryEl.innerHTML = `
                <div class="ratings-error">
                    <i class="fas fa-exclamation-triangle"></i>
                    <h3>Connection Error</h3>
                    <p>Unable to fetch ratings summary</p>
                </div>
            `;
        }
    }

    /**
     * Load top and bottom ratings
     */
    async loadTopBottomRatings() {
        const top10El = document.getElementById('top-10-ratings');
        const bottom10El = document.getElementById('bottom-10-ratings');
        
        try {
            const response = await fetch('/api/ratings/top_bottom');
            const data = await response.json();
            
            if (data.success) {
                const { top_10, bottom_10 } = data.data;
                
                // Display top 10
                top10El.innerHTML = this.renderRatingsList(top_10, 'top-performer');
                
                // Display bottom 10
                bottom10El.innerHTML = this.renderRatingsList(bottom_10, 'bottom-performer');
                
            } else {
                const errorHtml = `
                    <div class="ratings-error">
                        <i class="fas fa-exclamation-triangle"></i>
                        <h3>No Data Available</h3>
                        <p>${data.message || 'No ratings data found'}</p>
                    </div>
                `;
                top10El.innerHTML = errorHtml;
                bottom10El.innerHTML = errorHtml;
            }
        } catch (error) {
            const errorHtml = `
                <div class="ratings-error">
                    <i class="fas fa-exclamation-triangle"></i>
                    <h3>Connection Error</h3>
                    <p>Unable to fetch ratings data</p>
                </div>
            `;
            top10El.innerHTML = errorHtml;
            bottom10El.innerHTML = errorHtml;
        }
    }

    /**
     * Render ratings list
     */
    renderRatingsList(ratings, className = '') {
        if (!ratings || ratings.length === 0) {
            return `
                <div class="ratings-empty">
                    <i class="fas fa-chart-bar"></i>
                    <h3>No Data</h3>
                    <p>No ratings available at the moment</p>
                </div>
            `;
        }
        
        return `
            <div class="ratings-grid">
                ${ratings.map((rating, index) => this.renderRatingItem(rating, index + 1, className)).join('')}
            </div>
        `;
    }

    /**
     * Render individual rating item
     */
    renderRatingItem(rating, rank, className = '') {
        const score = rating.final_rating || 0;
        const scoreClass = score > 0 ? 'score-positive' : score < 0 ? 'score-negative' : 'score-neutral';
        const symbol = rating.trading_symbol || 'N/A';
        const ratingText = rating.rating_text || 'N/A';
        const emoji = rating.emoji || '⚪';
        
        // Format timestamp
        const lastUpdate = rating.timestamp ? new Date(rating.timestamp).toLocaleTimeString() : 'N/A';
        
        // Data quality indicators
        const dataQuality = rating.data_quality || {};
        const qualityScore = this.calculateDataQuality(dataQuality);
        const qualityClass = qualityScore >= 80 ? 'good' : qualityScore >= 60 ? 'fair' : 'poor';
        
        // Store rating data for click handler
        const ratingDataJson = JSON.stringify(rating).replace(/"/g, '&quot;');
        
        return `
            <div class="rating-item ${className}" onclick="window.dashboard.showStockDetail(${ratingDataJson})">
                <div class="stock-info">
                    <div class="stock-symbol">#${rank} ${symbol}</div>
                    <div class="stock-details">
                        <div class="rating-text">${ratingText}</div>
                        <div class="rating-details">
                            <div>Updated: ${lastUpdate}</div>
                            <div class="data-quality">
                                <span class="quality-indicator ${qualityClass}"></span>
                                Quality: ${qualityScore}%
                            </div>
                        </div>
                    </div>
                </div>
                <div class="rating-score">
                    <div class="rating-emoji">${emoji}</div>
                    <div class="score-value ${scoreClass}">${score.toFixed(2)}</div>
                </div>
            </div>
        `;
    }

    /**
     * Calculate data quality percentage
     */
    calculateDataQuality(dataQuality) {
        if (!dataQuality) return 0;
        
        const weights = {
            '1min_candles': 20,
            '3min_candles': 20,
            '5min_candles': 20,
            '15min_candles': 15,
            '30min_candles': 10,
            '60min_candles': 10,
            'daily_candles': 5
        };
        
        let totalScore = 0;
        let maxScore = 0;
        
        Object.entries(weights).forEach(([key, weight]) => {
            const candles = dataQuality[key] || 0;
            const expectedMin = key.includes('1min') || key.includes('3min') || key.includes('5min') ? 50 : 30;
            const score = Math.min(candles / expectedMin, 1) * weight;
            totalScore += score;
            maxScore += weight;
        });
        
        return Math.round((totalScore / maxScore) * 100);
    }

    /**
     * Refresh ratings manually
     */
    async refreshRatings() {
        const refreshBtn = document.getElementById('refresh-ratings-btn');
        
        if (refreshBtn) {
            // Disable button and show loading
            refreshBtn.disabled = true;
            refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Refreshing...';
            
            try {
                await this.loadLiveRatings();
                
                // Show success briefly
                refreshBtn.innerHTML = '<i class="fas fa-check"></i> Updated';
                setTimeout(() => {
                    refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh';
                    refreshBtn.disabled = false;
                }, 2000);
                
            } catch (error) {
                // Show error briefly
                refreshBtn.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Error';
                setTimeout(() => {
                    refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh';
                    refreshBtn.disabled = false;
                }, 2000);
            }
        }
    }

    /**
     * Start auto-refresh for ratings
     */
    startRatingsAutoRefresh() {
        // Clear any existing interval
        this.stopRatingsAutoRefresh();
        
        // Refresh every 30 seconds
        this.ratingsRefreshInterval = setInterval(() => {
            if (this.currentSection === 'live-ratings') {
                this.loadLiveRatings();
            }
        }, 30000);
        
        console.log('Started auto-refresh for live ratings');
    }

    /**
     * Stop auto-refresh for ratings
     */
    stopRatingsAutoRefresh() {
        if (this.ratingsRefreshInterval) {
            clearInterval(this.ratingsRefreshInterval);
            this.ratingsRefreshInterval = null;
        }
    }

    /**
     * Stock Detail Modal Functions
     */

    /**
     * Show stock detail modal
     */
    showStockDetail(stockData) {
        const overlay = document.getElementById('stock-detail-overlay');
        const symbolEl = document.getElementById('detail-stock-symbol');
        
        if (!overlay || !symbolEl) return;
        
        // Update modal header
        symbolEl.textContent = `${stockData.trading_symbol} - Stock Details`;
        
        // Store current stock data
        this.currentStockData = stockData;
        
        
        // Load content for all tabs
        this.loadStockOverview(stockData);
        this.loadRatingBreakdown(stockData);
        
        // Show modal
        overlay.classList.remove('hidden');
        setTimeout(() => {
            overlay.classList.add('active');
        }, 10);
        
        // Update panel averages
        this.updatePanelAverages();
    }

    /**
     * Close stock detail modal
     */
    closeStockDetail() {
        const overlay = document.getElementById('stock-detail-overlay');
        if (overlay) {
            overlay.classList.remove('active');
            setTimeout(() => {
                overlay.classList.add('hidden');
            }, 300);
        }
    }

    /**
     * Switch tabs in stock detail modal
     */
    switchDetailTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        
        // Update tab panels
        document.querySelectorAll('.tab-panel').forEach(panel => {
            panel.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');
        
    }

    /**
     * Load stock overview tab content
     */
    loadStockOverview(stockData) {
        const overviewEl = document.getElementById('stock-overview');
        
        const score = stockData.final_rating || 0;
        const compositeScore = stockData.composite_score || 0;
        const strategicScore = stockData.strategic_score || 0;
        const tacticalScore = stockData.tactical_score || 0;
        
        const scoreClass = score > 0 ? 'positive' : score < 0 ? 'negative' : 'neutral';
        const timestamp = new Date(stockData.timestamp).toLocaleString();
        
        overviewEl.innerHTML = `
            <div class="stock-overview-grid">
                <div class="overview-card ${scoreClass}">
                    <h4>Final Rating</h4>
                    <div class="value">${score.toFixed(2)}</div>
                </div>
                <div class="overview-card">
                    <h4>Rating Text</h4>
                    <div class="value">${stockData.rating_text || 'N/A'}</div>
                </div>
                <div class="overview-card">
                    <h4>Composite Score</h4>
                    <div class="value">${(compositeScore * 100).toFixed(2)}%</div>
                </div>
                <div class="overview-card">
                    <h4>Strategic Score</h4>
                    <div class="value">${(strategicScore * 100).toFixed(2)}%</div>
                </div>
                <div class="overview-card">
                    <h4>Tactical Score</h4>
                    <div class="value">${(tacticalScore * 100).toFixed(2)}%</div>
                </div>
                <div class="overview-card">
                    <h4>Last Updated</h4>
                    <div class="value">${timestamp}</div>
                </div>
            </div>
            
            <div class="breakdown-section">
                <h4>Data Quality Overview</h4>
                ${this.renderDataQualityDetails(stockData.data_quality || {})}
            </div>
        `;
    }


    /**
     * Load rating breakdown tab content
     */
    loadRatingBreakdown(stockData) {
        const breakdownEl = document.getElementById('rating-breakdown');
        
        const strategicContrib = stockData.strategic_contribution || 0;
        const tacticalContrib = stockData.tactical_contribution || 0;
        const finalRating = stockData.final_rating || 0;
        
        breakdownEl.innerHTML = `
            <div class="breakdown-section">
                <h4>Rating Composition</h4>
                <div class="breakdown-item">
                    <span class="breakdown-label">Strategic Contribution</span>
                    <span class="breakdown-value">${strategicContrib.toFixed(4)}</span>
                </div>
                <div class="breakdown-item">
                    <span class="breakdown-label">Tactical Contribution</span>
                    <span class="breakdown-value">${tacticalContrib.toFixed(4)}</span>
                </div>
                <div class="breakdown-item">
                    <span class="breakdown-label">Final Rating</span>
                    <span class="breakdown-value">${finalRating.toFixed(2)}</span>
                </div>
                
                <div class="breakdown-bar">
                    <div class="breakdown-bar-fill" style="width: ${Math.min(Math.abs(finalRating) * 10, 100)}%"></div>
                </div>
            </div>
            
            <div class="breakdown-section">
                <h4>Score Details</h4>
                <div class="breakdown-item">
                    <span class="breakdown-label">Strategic Score</span>
                    <span class="breakdown-value">${((stockData.strategic_score || 0) * 100).toFixed(2)}%</span>
                </div>
                <div class="breakdown-item">
                    <span class="breakdown-label">Tactical Score</span>
                    <span class="breakdown-value">${((stockData.tactical_score || 0) * 100).toFixed(2)}%</span>
                </div>
                <div class="breakdown-item">
                    <span class="breakdown-label">Composite Score</span>
                    <span class="breakdown-value">${((stockData.composite_score || 0) * 100).toFixed(2)}%</span>
                </div>
            </div>
            
            <div class="breakdown-section">
                <h4>Update Information</h4>
                <div class="breakdown-item">
                    <span class="breakdown-label">Last Strategic Update</span>
                    <span class="breakdown-value">${new Date(stockData.last_strategic_update).toLocaleString()}</span>
                </div>
                <div class="breakdown-item">
                    <span class="breakdown-label">Rating Timestamp</span>
                    <span class="breakdown-value">${new Date(stockData.timestamp).toLocaleString()}</span>
                </div>
            </div>
        `;
    }

    /**
     * Render data quality details
     */
    renderDataQualityDetails(dataQuality) {
        const qualityItems = Object.entries(dataQuality).map(([timeframe, count]) => {
            const quality = count > 100 ? 'good' : count > 50 ? 'fair' : 'poor';
            return `
                <div class="breakdown-item">
                    <span class="breakdown-label">${timeframe.replace('_', ' ')}</span>
                    <span class="breakdown-value">
                        <span class="quality-indicator ${quality}"></span>
                        ${count} candles
                    </span>
                </div>
            `;
        }).join('');

        return `<div>${qualityItems}</div>`;
    }

    /**
     * Update panel average scores
     */
    async updatePanelAverages() {
        try {
            const response = await fetch('/api/ratings/top_bottom');
            const data = await response.json();
            
            if (data.success) {
                const { top_10, bottom_10 } = data.data;
                
                // Calculate averages
                const topAvg = top_10.reduce((sum, stock) => sum + (stock.final_rating || 0), 0) / top_10.length;
                const bottomAvg = bottom_10.reduce((sum, stock) => sum + (stock.final_rating || 0), 0) / bottom_10.length;
                
                // Update badges
                const topBadge = document.getElementById('top-avg-score');
                const bottomBadge = document.getElementById('bottom-avg-score');
                
                if (topBadge) {
                    topBadge.textContent = `Avg: ${topAvg.toFixed(2)}`;
                }
                
                if (bottomBadge) {
                    bottomBadge.textContent = `Avg: ${bottomAvg.toFixed(2)}`;
                }
            }
        } catch (error) {
            console.error('Error updating panel averages:', error);
        }
    }

    /**
     * Enhanced load top bottom ratings with panel averages
     */
    async loadTopBottomRatings() {
        const top10El = document.getElementById('top-10-ratings');
        const bottom10El = document.getElementById('bottom-10-ratings');
        
        try {
            const response = await fetch('/api/ratings/top_bottom');
            const data = await response.json();
            
            if (data.success) {
                const { top_10, bottom_10 } = data.data;
                
                // Display top 10
                top10El.innerHTML = this.renderRatingsList(top_10, 'top-performer');
                
                // Display bottom 10
                bottom10El.innerHTML = this.renderRatingsList(bottom_10, 'bottom-performer');
                
                // Update panel averages
                this.updatePanelAverages();
                
            } else {
                const errorHtml = `
                    <div class="ratings-error">
                        <i class="fas fa-exclamation-triangle"></i>
                        <h3>No Data Available</h3>
                        <p>${data.message || 'No ratings data found'}</p>
                    </div>
                `;
                top10El.innerHTML = errorHtml;
                bottom10El.innerHTML = errorHtml;
            }
        } catch (error) {
            const errorHtml = `
                <div class="ratings-error">
                    <i class="fas fa-exclamation-triangle"></i>
                    <h3>Connection Error</h3>
                    <p>Unable to fetch ratings data</p>
                </div>
            `;
            top10El.innerHTML = errorHtml;
            bottom10El.innerHTML = errorHtml;
        }
    }

    /**
     * Cleanup when dashboard is destroyed
     */
    destroy() {
        if (this.loginStatusInterval) {
            clearInterval(this.loginStatusInterval);
        }
        if (this.connectionStatusInterval) {
            clearInterval(this.connectionStatusInterval);
        }
        if (this.ratingsRefreshInterval) {
            clearInterval(this.ratingsRefreshInterval);
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new TradingDashboard();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.dashboard) {
        window.dashboard.destroy();
    }
});
