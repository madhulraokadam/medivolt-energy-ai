// MediVolt Login Script - Dynamic API URL for local and production
const API_BASE = window.location.origin;

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('loginForm');
    if (!form) return;
    
    const username = document.getElementById('username');
    const password = document.getElementById('password');
    const button = form.querySelector('button');

    // Validation functions
    function showError(input, message) {
        const formGroup = input.parentElement;
        formGroup.classList.add('has-error');
        input.classList.add('error');
        
        let errorEl = formGroup.querySelector('.error-message');
        if (!errorEl) {
            errorEl = document.createElement('div');
            errorEl.className = 'error-message';
            formGroup.appendChild(errorEl);
        }
        errorEl.textContent = message;
    }

    function clearError(input) {
        const formGroup = input.parentElement;
        formGroup.classList.remove('has-error');
        input.classList.remove('error');
    }

    function validate() {
        let valid = true;
        
        clearError(username);
        clearError(password);

        if (!username.value || username.value.length < 1) {
            showError(username, 'Please enter a username');
            valid = false;
        }

        if (!password.value || password.value.length < 1) {
            showError(password, 'Please enter a password');
            valid = false;
        }

        return valid;
    }

    // Real-time validation
    [username, password].forEach(input => {
        input.addEventListener('input', () => clearError(input));
    });

    // Form submission
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!validate()) return;

        button.textContent = 'Signing In...';
        button.disabled = true;

        // Call backend API for login - using absolute URL
        const loginData = {
            username: username.value,
            password: password.value
        };

        fetch(API_BASE + '/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(loginData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Store user info in localStorage
                localStorage.setItem('user', JSON.stringify(data.user));
                localStorage.setItem('loggedIn', 'true');

                // Redirect to dashboard
                window.location.href = 'dashboard.html';
            } else {
                // Show error message
                showError(password, data.message || 'Login failed. Please try again.');
                button.textContent = 'Sign In';
                button.disabled = false;
            }
        })
        .catch(error => {
            showError(password, 'Connection error. Please check your internet.');
            button.textContent = 'Sign In';
            button.disabled = false;
            console.error('Login error:', error);
        });
    });
});
