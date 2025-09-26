// Credit Risk Scorecard UI JavaScript

class CreditRiskUI {
    constructor() {
        this.form = document.getElementById('loanForm');
        this.results = document.getElementById('results');
        this.loading = document.getElementById('loading');
        this.submitBtn = document.getElementById('submitBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.closeResultsBtn = document.getElementById('closeResults');
        
        this.init();
    }
    
    init() {
        // Event listeners
        this.form.addEventListener('submit', (e) => this.handleSubmit(e));
        this.clearBtn.addEventListener('click', () => this.clearForm());
        this.closeResultsBtn.addEventListener('click', () => this.hideResults());
        
        // Add input validation
        this.addInputValidation();
        
        // Auto-calculate loan burden if loan amount is provided
        this.addLoanAmountField();
    }
    
    addInputValidation() {
        const inputs = this.form.querySelectorAll('input[type="number"]');
        inputs.forEach(input => {
            input.addEventListener('blur', () => this.validateInput(input));
            input.addEventListener('input', () => this.clearInputError(input));
        });
    }
    
    addLoanAmountField() {
        // Add loan amount field dynamically
        const purposeGroup = document.querySelector('#purpose').closest('.form-group');
        const loanAmountGroup = document.createElement('div');
        loanAmountGroup.className = 'form-group';
        loanAmountGroup.innerHTML = `
            <label for="loan_amount">
                <i class="fas fa-dollar-sign"></i> Loan Amount ($)
            </label>
            <input type="number" id="loan_amount" name="loan_amount" 
                   placeholder="e.g., 15000" min="0" step="1000">
            <small>Amount you want to borrow</small>
        `;
        
        purposeGroup.parentNode.insertBefore(loanAmountGroup, purposeGroup.nextSibling);
        
        // Auto-calculate loan burden
        const loanAmountInput = document.getElementById('loan_amount');
        const annualIncInput = document.getElementById('annual_inc');
        
        const calculateLoanBurden = () => {
            const loanAmount = parseFloat(loanAmountInput.value) || 0;
            const annualInc = parseFloat(annualIncInput.value) || 0;
            
            if (loanAmount > 0 && annualInc > 0) {
                const burden = loanAmount / (annualInc + 1);
                const burdenInput = document.getElementById('loan_burden');
                if (burdenInput) {
                    burdenInput.value = burden.toFixed(4);
                }
            }
        };
        
        loanAmountInput.addEventListener('input', calculateLoanBurden);
        annualIncInput.addEventListener('input', calculateLoanBurden);
    }
    
    validateInput(input) {
        const value = parseFloat(input.value);
        const min = parseFloat(input.min) || 0;
        const max = parseFloat(input.max) || Infinity;
        
        if (input.value && (value < min || value > max)) {
            this.showInputError(input, `Value must be between ${min} and ${max}`);
            return false;
        }
        
        this.clearInputError(input);
        return true;
    }
    
    showInputError(input, message) {
        this.clearInputError(input);
        
        const errorDiv = document.createElement('div');
        errorDiv.className = 'input-error';
        errorDiv.style.color = '#e74c3c';
        errorDiv.style.fontSize = '0.8rem';
        errorDiv.style.marginTop = '0.3rem';
        errorDiv.textContent = message;
        
        input.style.borderColor = '#e74c3c';
        input.parentNode.appendChild(errorDiv);
    }
    
    clearInputError(input) {
        input.style.borderColor = '#e1e8ed';
        const errorDiv = input.parentNode.querySelector('.input-error');
        if (errorDiv) {
            errorDiv.remove();
        }
    }
    
    async handleSubmit(e) {
        e.preventDefault();
        
        // Validate all inputs
        const isValid = this.validateForm();
        if (!isValid) {
            this.showNotification('Please fix the errors in the form', 'error');
            return;
        }
        
        // Show loading
        this.showLoading();
        
        try {
            // Collect form data
            const formData = this.collectFormData();
            
            // Make API request
            const response = await fetch('/api/v1/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Request failed');
            }
            
            const result = await response.json();
            
            // Display results
            this.displayResults(result);
            
        } catch (error) {
            console.error('Error:', error);
            this.showNotification(`Error: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    validateForm() {
        const requiredFields = ['annual_inc', 'int_rate', 'credit_history_length', 'purpose', 'verification_status'];
        let isValid = true;
        
        requiredFields.forEach(fieldName => {
            const field = document.getElementById(fieldName);
            if (!field.value.trim()) {
                this.showInputError(field, 'This field is required');
                isValid = false;
            }
        });
        
        // Validate number inputs
        const numberInputs = this.form.querySelectorAll('input[type="number"]');
        numberInputs.forEach(input => {
            if (input.value && !this.validateInput(input)) {
                isValid = false;
            }
        });
        
        return isValid;
    }
    
    collectFormData() {
        const formData = new FormData(this.form);
        const data = {};
        
        for (let [key, value] of formData.entries()) {
            if (value.trim()) {
                // Convert numbers
                if (['annual_inc', 'int_rate', 'credit_history_length', 'total_rev_hi_lim', 
                     'tot_cur_bal', 'revol_util', 'loan_amount'].includes(key)) {
                    data[key] = parseFloat(value);
                } else {
                    data[key] = value;
                }
            }
        }
        
        // Calculate loan burden if not provided
        if (!data.loan_burden && data.loan_amount && data.annual_inc) {
            data.loan_burden = data.loan_amount / (data.annual_inc + 1);
        }
        
        return data;
    }
    
    displayResults(result) {
        // Update result elements
        document.getElementById('creditScore').textContent = result.credit_score;
        document.getElementById('defaultProb').textContent = (result.default_probability * 100).toFixed(2) + '%';
        document.getElementById('riskLevel').textContent = result.risk_level;
        
        // Update result message
        const messageEl = document.getElementById('resultMessage');
        if (result.message) {
            messageEl.textContent = result.message;
        }
        
        // Style risk level
        const riskEl = document.getElementById('riskLevel');
        riskEl.className = 'risk-value';
        
        if (result.risk_level.includes('Low')) {
            riskEl.style.background = '#27ae60';
            riskEl.style.color = 'white';
        } else if (result.risk_level.includes('Medium')) {
            riskEl.style.background = '#f39c12';
            riskEl.style.color = 'white';
        } else {
            riskEl.style.background = '#e74c3c';
            riskEl.style.color = 'white';
        }
        
        // Show results
        this.showResults();
        
        // Show success notification
        this.showNotification('Credit score calculated successfully!', 'success');
    }
    
    showResults() {
        this.results.classList.remove('hidden');
        this.results.scrollIntoView({ behavior: 'smooth' });
    }
    
    hideResults() {
        this.results.classList.add('hidden');
    }
    
    clearForm() {
        this.form.reset();
        this.hideResults();
        
        // Clear any error states
        const inputs = this.form.querySelectorAll('input, select');
        inputs.forEach(input => this.clearInputError(input));
        
        this.showNotification('Form cleared', 'info');
    }
    
    showLoading() {
        this.loading.classList.remove('hidden');
        this.submitBtn.disabled = true;
        this.submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Calculating...';
    }
    
    hideLoading() {
        this.loading.classList.add('hidden');
        this.submitBtn.disabled = false;
        this.submitBtn.innerHTML = '<i class="fas fa-calculator"></i> Calculate Credit Score';
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            z-index: 1001;
            animation: slideInRight 0.3s ease-out;
            max-width: 300px;
        `;
        
        // Set background color based on type
        const colors = {
            success: '#27ae60',
            error: '#e74c3c',
            info: '#3498db',
            warning: '#f39c12'
        };
        notification.style.background = colors[type] || colors.info;
        
        notification.textContent = message;
        
        // Add to DOM
        document.body.appendChild(notification);
        
        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease-in';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Initialize the UI when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new CreditRiskUI();
});

// Add some sample data functionality
function fillSampleData() {
    const sampleData = {
        annual_inc: 75000,
        int_rate: 12.5,
        credit_history_length: 5.5,
        purpose: 'debt_consolidation',
        total_rev_hi_lim: 25000,
        tot_cur_bal: 5000,
        revol_util: 30,
        verification_status: 'Verified',
        loan_amount: 15000
    };
    
    Object.entries(sampleData).forEach(([key, value]) => {
        const element = document.getElementById(key);
        if (element) {
            element.value = value;
        }
    });
    
    // Trigger loan burden calculation
    const loanAmountInput = document.getElementById('loan_amount');
    if (loanAmountInput) {
        loanAmountInput.dispatchEvent(new Event('input'));
    }
}

// Add sample data button (for demo purposes)
document.addEventListener('DOMContentLoaded', () => {
    const sampleBtn = document.createElement('button');
    sampleBtn.type = 'button';
    sampleBtn.className = 'btn-secondary';
    sampleBtn.innerHTML = '<i class="fas fa-flask"></i> Fill Sample Data';
    sampleBtn.onclick = fillSampleData;
    
    const formActions = document.querySelector('.form-actions');
    if (formActions) {
        formActions.appendChild(sampleBtn);
    }
});
