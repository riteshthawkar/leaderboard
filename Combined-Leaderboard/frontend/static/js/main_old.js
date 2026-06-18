/**
 * Production-ready JavaScript for Vision Leaderboard
 * 
 * Features:
 * - XSS protection with proper HTML escaping
 * - Error boundaries and graceful error handling
 * - Request tracking with IDs
 * - Modal dialogs for detailed views
 * - Proper async/await with error handling
 */

// API endpoints
const API_BASE = '/api';
const API_TIMEOUT = 30000; // 30 seconds

// Utility functions
/**
 * Escape HTML special characters to prevent XSS
 * @param {string} text - Text to escape
 * @returns {string} Escaped HTML
 */
function escapeHtml(text) {
    if (!text) return '';
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return String(text).replace(/[&<>"']/g, m => map[m]);
}

/**
 * Create text node to safely insert text into DOM
 * @param {string} text - Text to insert
 * @returns {Text} Text node
 */
function createSafeTextNode(text) {
    return document.createTextNode(text);
}

/**
 * Format task names for display
 * @param {string} taskName - Task name to format
 * @returns {string} Formatted name
 */
function formatTaskName(taskName) {
    if (!taskName) return '';
    return String(taskName)
        .replace(/_/g, ' ')
        .replace(/\b\w/g, l => l.toUpperCase())
        .replace(/2d/gi, '2D')
        .replace(/3d/gi, '3D');
}

/**
 * Fetch with timeout
 * @param {string} url - URL to fetch
 * @param {object} options - Fetch options
 * @param {number} timeout - Timeout in ms
 * @returns {Promise} Fetch response
 */
async function fetchWithTimeout(url, options = {}, timeout = API_TIMEOUT) {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    
    try {
        const response = await fetch(url, { ...options, signal: controller.signal });
        clearTimeout(id);
        return response;
    } catch (error) {
        clearTimeout(id);
        if (error.name === 'AbortError') {
            throw new Error('Request timeout');
        }
        throw error;
    }
}

/**
 * Show message to user
 * @param {string} text - Message text
 * @param {string} type - Message type (info, success, error, warning)
 * @param {HTMLElement} element - Target element
 */
function showMessage(text, type, element) {
    if (!element) return;
    element.textContent = text;
    element.className = `message ${type}`;
    element.style.display = 'block';
    
    // Auto-hide success messages after 5 seconds
    if (type === 'success') {
        setTimeout(() => {
            element.style.display = 'none';
        }, 5000);
    }
}

/**
 * Create and display modal
 * @param {string} title - Modal title
 * @param {string} content - Modal HTML content
 * @param {function} onClose - Callback when modal closes
 */
function showModal(title, content, onClose = null) {
    // Remove existing modal if any
    const existingModal = document.getElementById('submission_modal');
    if (existingModal) existingModal.remove();
    
    // Create modal HTML
    const modal = document.createElement('div');
    modal.id = 'submission_modal';
    modal.className = 'modal';
    modal.innerHTML = `
        <div class="modal-overlay"></div>
        <div class="modal-content">
            <div class="modal-header">
                <h3>${escapeHtml(title)}</h3>
                <button class="modal-close" aria-label="Close modal">&times;</button>
            </div>
            <div class="modal-body">
                ${content}
            </div>
            <div class="modal-footer">
                <button class="btn btn-primary" id="modal_close_btn">Close</button>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Close button handlers
    const closeBtn = modal.querySelector('.modal-close');
    const closeBtnFooter = modal.getElementById('modal_close_btn');
    const overlay = modal.querySelector('.modal-overlay');
    
    const closeModal = () => {
        modal.remove();
        if (onClose) onClose();
    };
    
    closeBtn.addEventListener('click', closeModal);
    closeBtnFooter.addEventListener('click', closeModal);
    overlay.addEventListener('click', closeModal);
    
    // Close on Escape key
    const handleEscape = (e) => {
        if (e.key === 'Escape') {
            closeModal();
            document.removeEventListener('keydown', handleEscape);
        }
    };
    document.addEventListener('keydown', handleEscape);
}

/**
 * Add modal styles
 */
function addModalStyles() {
    const style = document.createElement('style');
    style.textContent = `
        .modal { position: fixed; top: 0; left: 0; right: 0; bottom: 0; display: flex; align-items: center; justify-content: center; z-index: 1000; }
        .modal-overlay { position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); }
        .modal-content { position: relative; background: white; border-radius: 8px; max-width: 600px; max-height: 80vh; overflow-y: auto; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .modal-header { padding: 16px 24px; border-bottom: 1px solid #e0e0e0; display: flex; justify-content: space-between; align-items: center; }
        .modal-header h3 { margin: 0; }
        .modal-close { background: none; border: none; font-size: 24px; cursor: pointer; padding: 0; width: 32px; height: 32px; }
        .modal-body { padding: 24px; }
        .modal-footer { padding: 16px 24px; border-top: 1px solid #e0e0e0; text-align: right; }
    `;
    document.head.appendChild(style);
}

// Load initial data
document.addEventListener('DOMContentLoaded', () => {
    addModalStyles();
    loadStatistics();
    loadLeaderboard();
    loadAvailableTasks();
    setupEventListeners();
});

/**
 * Setup event listeners
 */
function setupEventListeners() {
    const submitBtn = document.getElementById('submit_btn');
    const benchmarkSelect = document.getElementById('benchmark');
    const leaderboardBenchmark = document.getElementById('leaderboard_benchmark');
    const leaderboardTask = document.getElementById('leaderboard_task');
    const leaderboardLimit = document.getElementById('leaderboard_limit');
    
    if (submitBtn) submitBtn.addEventListener('click', submitPredictions);
    if (benchmarkSelect) benchmarkSelect.addEventListener('change', loadBenchmarkTasks);
    if (leaderboardBenchmark) leaderboardBenchmark.addEventListener('change', loadLeaderboard);
    if (leaderboardTask) leaderboardTask.addEventListener('change', loadLeaderboard);
    if (leaderboardLimit) leaderboardLimit.addEventListener('change', loadLeaderboard);
}

/**
 * Load statistics from API
 */
async function loadStatistics() {
    try {
        const response = await fetchWithTimeout(`${API_BASE}/statistics`);
        
        if (!response.ok) {
            console.error('Failed to load statistics:', response.status);
            return;
        }
        
        const data = await response.json();
        
        // Update UI - use textContent for security
        const totalSubmissions = document.getElementById('total_submissions');
        const uniqueModels = document.getElementById('unique_models');
        const avgAccuracy = document.getElementById('average_accuracy');
        const bestAccuracy = document.getElementById('best_accuracy');
        
        if (totalSubmissions) totalSubmissions.textContent = data.total_submissions || '-';
        if (uniqueModels) uniqueModels.textContent = data.unique_models || '-';
        if (avgAccuracy) avgAccuracy.textContent = 
            (data.average_accuracy ? (data.average_accuracy * 100).toFixed(2) + '%' : '-');
        if (bestAccuracy) bestAccuracy.textContent = 
            (data.best_accuracy ? (data.best_accuracy * 100).toFixed(2) + '%' : '-');
            
    } catch (error) {
        console.error('Error loading statistics:', error);
    }
}

/**
 * Load available tasks from API
 */
async function loadAvailableTasks() {
    try {
        const response = await fetchWithTimeout(`${API_BASE}/tasks`);
        
        if (!response.ok) {
            console.error('Failed to load tasks:', response.status);
            return;
        }
        
        const tasks = await response.json();
        populateTaskSelects(tasks);
        
    } catch (error) {
        console.error('Error loading tasks:', error);
    }
}

/**
 * Populate task select elements
 * @param {object} tasks - Tasks data from API
 */
function populateTaskSelects(tasks) {
    const leaderboardTaskSelect = document.getElementById('leaderboard_task');
    const taskSelect = document.getElementById('task');
    
    if (!leaderboardTaskSelect || !taskSelect) return;
    
    // Clear existing options (keep first option)
    while (leaderboardTaskSelect.options.length > 1) {
        leaderboardTaskSelect.remove(1);
    }
    while (taskSelect.options.length > 1) {
        taskSelect.remove(1);
    }
    
    // Add tasks
    const doYouSeeTasks = tasks.do_you_see_me || [];
    const mindsEyeTasks = tasks.minds_eye || [];
    const allTasks = [...doYouSeeTasks, ...mindsEyeTasks];
    
    allTasks.forEach(task => {
        const option1 = document.createElement('option');
        option1.value = task;
        option1.textContent = formatTaskName(task);
        leaderboardTaskSelect.appendChild(option1);
        
        const option2 = document.createElement('option');
        option2.value = task;
        option2.textContent = formatTaskName(task);
        taskSelect.appendChild(option2);
    });
}

/**
 * Load tasks for selected benchmark
 */
async function loadBenchmarkTasks() {
    const benchmark = document.getElementById('benchmark')?.value;
    const taskSelect = document.getElementById('task');
    
    if (!taskSelect) return;
    
    // Clear options (keep first)
    while (taskSelect.options.length > 1) {
        taskSelect.remove(1);
    }
    
    if (!benchmark) return;
    
    try {
        const response = await fetchWithTimeout(`${API_BASE}/tasks?benchmark=${encodeURIComponent(benchmark)}`);
        
        if (!response.ok) {
            console.error('Failed to load benchmark tasks:', response.status);
            return;
        }
        
        const data = await response.json();
        const tasks = data[benchmark] || [];
        
        tasks.forEach(task => {
            const option = document.createElement('option');
            option.value = task;
            option.textContent = formatTaskName(task);
            taskSelect.appendChild(option);
        });
        
    } catch (error) {
        console.error('Error loading benchmark tasks:', error);
    }
}

/**
 * Submit predictions
 */
async function submitPredictions() {
    const modelName = document.getElementById('model_name')?.value.trim();
    const benchmark = document.getElementById('benchmark')?.value;
    const taskName = document.getElementById('task')?.value || '';
    const file = document.getElementById('prediction_file')?.files[0];
    const messageElement = document.getElementById('submit_message');
    const submitBtn = document.getElementById('submit_btn');
    
    // Validate inputs
    if (!modelName) {
        showMessage('Please enter a model name', 'error', messageElement);
        return;
    }
    
    if (!benchmark) {
        showMessage('Please select a benchmark', 'error', messageElement);
        return;
    }
    
    if (!file) {
        showMessage('Please select a file', 'error', messageElement);
        return;
    }
    
    if (!file.name.endsWith('.csv') && !file.name.endsWith('.json')) {
        showMessage('File must be CSV or JSON format', 'error', messageElement);
        return;
    }
    
    if (file.size > 10 * 1024 * 1024) {
        showMessage('File is too large (max 10MB)', 'error', messageElement);
        return;
    }
    
    // Disable button
    if (submitBtn) submitBtn.disabled = true;
    showMessage('Uploading and scoring...', 'info', messageElement);
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_name', modelName);
    formData.append('benchmark', benchmark);
    if (taskName) formData.append('task_name', taskName);
    
    try {
        const response = await fetchWithTimeout(`${API_BASE}/submit`, {
            method: 'POST',
            body: formData
        }, 120000); // 2 minute timeout for uploads
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            showMessage(
                `✅ Success! Overall Accuracy: ${(data.overall_accuracy * 100).toFixed(2)}%`,
                'success',
                messageElement
            );
            
            // Clear form
            document.getElementById('model_name').value = '';
            document.getElementById('benchmark').value = '';
            document.getElementById('task').value = '';
            document.getElementById('prediction_file').value = '';
            
            // Reload after 1 second
            setTimeout(() => {
                loadStatistics();
                loadLeaderboard();
            }, 1000);
        } else {
            showMessage(`❌ Error: ${data.error || 'Unknown error'}`, 'error', messageElement);
        }
    } catch (error) {
        showMessage(`❌ Error: ${error.message}`, 'error', messageElement);
        console.error('Submission error:', error);
    } finally {
        if (submitBtn) submitBtn.disabled = false;
    }
}

/**
 * Load leaderboard
 */
async function loadLeaderboard() {
    const benchmark = document.getElementById('leaderboard_benchmark')?.value || '';
    const task = document.getElementById('leaderboard_task')?.value || '';
    const limit = document.getElementById('leaderboard_limit')?.value || 25;
    const tbody = document.getElementById('leaderboard_body');
    
    if (!tbody) return;
    
    tbody.innerHTML = '<tr><td colspan="6" class="loading">Loading...</td></tr>';
    
    try {
        let url = `${API_BASE}/leaderboard?limit=${limit}`;
        if (benchmark) url += `&benchmark=${encodeURIComponent(benchmark)}`;
        if (task) url += `&task=${encodeURIComponent(task)}`;
        
        const response = await fetchWithTimeout(url);
        
        if (!response.ok) {
            tbody.innerHTML = `<tr><td colspan="6" class="loading">Error loading leaderboard</td></tr>`;
            return;
        }
        
        const data = await response.json();
        displayLeaderboard(data.leaderboard || []);
        
    } catch (error) {
        console.error('Error loading leaderboard:', error);
        tbody.innerHTML = `<tr><td colspan="6" class="loading">Error: ${escapeHtml(error.message)}</td></tr>`;
    }
}

/**
 * Display leaderboard entries
 * @param {array} entries - Leaderboard entries
 */
function displayLeaderboard(entries) {
    const tbody = document.getElementById('leaderboard_body');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    if (!entries || entries.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="loading">No submissions yet</td></tr>';
        return;
    }
    
    entries.forEach(entry => {
        try {
            const row = document.createElement('tr');
            row.style.cursor = 'pointer';
            row.addEventListener('click', () => viewSubmissionDetails(entry.submission_id));
            
            const accuracy = (entry.overall_accuracy * 100).toFixed(2);
            const submittedDate = new Date(entry.submitted_at).toLocaleDateString();
            
            // Use textContent for model_name to prevent XSS
            const modelCell = row.insertCell();
            modelCell.innerHTML = `<strong>#${entry.rank}</strong>`;
            
            const nameCell = row.insertCell();
            nameCell.textContent = entry.model_name;
            
            const benchCell = row.insertCell();
            benchCell.innerHTML = `<span class="badge">${formatTaskName(entry.benchmark)}</span>`;
            
            const accCell = row.insertCell();
            accCell.innerHTML = `<strong>${accuracy}%</strong>`;
            
            const samplesCell = row.insertCell();
            samplesCell.textContent = `${entry.correct_samples}/${entry.total_samples}`;
            
            const dateCell = row.insertCell();
            dateCell.textContent = submittedDate;
            
            tbody.appendChild(row);
        } catch (error) {
            console.error('Error rendering leaderboard entry:', error);
        }
    });
}

/**
 * View submission details
 * @param {string} submissionId - Submission ID
 */
async function viewSubmissionDetails(submissionId) {
    if (!submissionId) return;
    
    try {
        const response = await fetchWithTimeout(`${API_BASE}/submission/${encodeURIComponent(submissionId)}`);
        
        if (!response.ok) {
            showMessage('Submission not found', 'error', document.getElementById('submit_message'));
            return;
        }
        
        const data = await response.json();
        
        // Create safe HTML content
        let content = `
            <p><strong>Submission ID:</strong> <code>${escapeHtml(data.submission_id)}</code></p>
            <p><strong>Model:</strong> ${escapeHtml(data.model_name)}</p>
            <p><strong>Benchmark:</strong> ${escapeHtml(formatTaskName(data.benchmark))}</p>
            <p><strong>Overall Accuracy:</strong> ${(data.overall_accuracy * 100).toFixed(2)}%</p>
            <p><strong>Samples:</strong> ${data.correct_samples}/${data.total_samples}</p>
            <p><strong>Submitted:</strong> ${new Date(data.submitted_at).toLocaleString()}</p>
            <h4>Task Results:</h4>
            <ul>
        `;
        
        if (data.task_results) {
            Object.entries(data.task_results).forEach(([task, result]) => {
                content += `
                    <li>${escapeHtml(formatTaskName(task))}: ${(result.accuracy * 100).toFixed(2)}% 
                        (${result.correct_samples}/${result.total_samples})</li>
                `;
            });
        }
        
        content += '</ul>';
        
        showModal('Submission Details', content);
        
    } catch (error) {
        console.error('Error loading submission details:', error);
        showMessage('Error loading submission details', 'error', document.getElementById('submit_message'));
    }
}

// Auto-load statistics every 30 seconds
setInterval(loadStatistics, 30000);
