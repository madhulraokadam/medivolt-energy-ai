// MediVolt Dashboard JavaScript - Dynamic API URL for local and production
const API_BASE = window.location.origin + '/api';

document.addEventListener('DOMContentLoaded', function() {
    
    // Check ML API status on load
    checkMLStatus();
    
    // Load initial data
    loadDashboardData();

    // Chart instances
    let energyChart = null;
    let energyTrendChart = null;
    let hvacChart = null;
    let hvacTrendChart = null;
    let carbonChart = null;
    let realtimeChart = null;

    // Initialize charts if canvas elements exist
    function initCharts() {
        try {
            const eCtx = document.getElementById('energyChart');
            if (eCtx) {
                energyChart = new Chart(eCtx, {
                    type: 'bar',
                    data: {
                        labels: ['Low', 'Avg', 'High'],
                        datasets: [{
                            label: 'kWh',
                            data: [1800, 2300, 2700],
                            backgroundColor: ['#60A5FA', '#34D399', '#F97316']
                        }]
                    },
                    options: { responsive: true, maintainAspectRatio: false }
                });
            }

            const eTrend = document.getElementById('energyTrendChart');
            if (eTrend) {
                energyTrendChart = new Chart(eTrend, {
                    type: 'line',
                    data: { labels: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], datasets:[{ label:'kWh', data:[2500,2600,2400,2700,2800,2000,1900], borderColor:'#06b6d4', backgroundColor:'rgba(6,182,212,0.1)', tension:0.3 }]},
                    options: { responsive:true, maintainAspectRatio:false }
                });
            }

            const hCtx = document.getElementById('hvacChart');
            if (hCtx) {
                hvacChart = new Chart(hCtx, {
                    type: 'bar',
                    data: { labels:['Current','Optimized','Potential'], datasets:[{ label:'%', data:[85,55,35], backgroundColor:['#f87171','#60a5fa','#34d399'] }]},
                    options:{ responsive:true, maintainAspectRatio:false }
                });
            }

            const hTrend = document.getElementById('hvacTrendChart');
            if (hTrend) {
                hvacTrendChart = new Chart(hTrend, { type:'bar', data:{ labels:['W1','W2','W3','W4'], datasets:[{ label:'Efficiency', data:[80,72,65,58], backgroundColor:'#60a5fa' }]}, options:{ responsive:true, maintainAspectRatio:false } });
            }

            const cCtx = document.getElementById('carbonChart');
            if (cCtx) {
                carbonChart = new Chart(cCtx, { type:'doughnut', data:{ labels:['Grid','Solar','Other'], datasets:[{ data:[70,20,10], backgroundColor:['#f59e0b','#34d399','#93c5fd'] }]}, options:{ responsive:true, maintainAspectRatio:false } });
            }

            const rCtx = document.getElementById('realtimeChart');
            if (rCtx) {
                realtimeChart = new Chart(rCtx, { type:'line', data:{ labels:[], datasets:[{ label:'kW', data:[], borderColor:'#ef4444', backgroundColor:'rgba(239,68,68,0.08)', tension:0.3 }]}, options:{ responsive:true, maintainAspectRatio:false, scales:{ x:{ display:true }, y:{ beginAtZero:true } } } });
            }
        } catch (err) {
            console.warn('Chart init error', err);
        }
    }

    // initialize charts
    initCharts();
    
    // Navigation handling
    const navItems = document.querySelectorAll('.nav-item[data-page]');
    const pages = document.querySelectorAll('.page-content');
    const headerTitle = document.querySelector('.header-left h1');

    const pageTitles = {
        'dashboard': 'Energy Dashboard',
        'realtime': 'Real-time Monitor',
        'department': 'Department Analysis',
        'cost': 'Cost Analysis',
        'alerts': 'Alert System',
        'weather': 'Weather Impact',
        'energy': 'Energy Prediction',
        'hvac': 'HVAC Optimization',
        'carbon': 'Carbon Forecast',
        'reports': 'Reports',
        'settings': 'Settings'
    };

    navItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            navItems.forEach(nav => nav.classList.remove('active'));
            this.classList.add('active');
            const pageName = this.getAttribute('data-page');
            pages.forEach(page => page.classList.remove('active'));
            document.getElementById('page-' + pageName).classList.add('active');
            headerTitle.textContent = pageTitles[pageName] || 'Dashboard';
        });
    });

    // Feature card navigation
    const featureCards = document.querySelectorAll('.feature-card');
    featureCards.forEach(card => {
        card.addEventListener('click', function() {
            const feature = this.getAttribute('data-feature');
            navItems.forEach(nav => nav.classList.remove('active'));
            document.querySelector(`.nav-item[data-page="${feature}"]`).classList.add('active');
            pages.forEach(page => page.classList.remove('active'));
            document.getElementById('page-' + feature).classList.add('active');
            headerTitle.textContent = pageTitles[feature];
        });
    });

    // ==================== API FUNCTIONS ====================
    
    // Check ML Model Status
    async function checkMLStatus() {
        try {
            const response = await fetch(`${API_BASE}/ml/status`);
            const data = await response.json();
            console.log('ML Status:', data);
            if (data.available) {
                console.log('✅ ML Models connected and ready!');
            }
        } catch (error) {
            console.error('❌ ML API not available. Make sure Flask server is running on port 5000');
        }
    }

    // Load Dashboard Data from ML APIs
    async function loadDashboardData() {
        // Load real-time data
        updateRealTime();
        
        // Load energy prediction for dashboard
        try {
            const energyResponse = await fetch(`${API_BASE}/ml/predict/energy`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    num_beds: 200,
                    equipment_hours: 12,
                    outdoor_temp: 25,
                    humidity: 60,
                    building_area: 15000,
                    day_of_week: new Date().getDay(),
                    is_weekend: (new Date().getDay() === 0 || new Date().getDay() === 6) ? 1 : 0,
                    occupancy_rate: 0.8,
                    hvac_efficiency: 0.75
                })
            });
            const energyData = await energyResponse.json();
            if (energyData.success) {
                updateEnergyDashboard(energyData.prediction);
            }
        } catch (e) {
            console.log('Using fallback data');
        }
        
        // Load carbon data
        try {
            const carbonResponse = await fetch(`${API_BASE}/ml/predict/carbon`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    energy_consumption: 50000,
                    energy_source: 0,
                    grid_mix_percentage: 70,
                    solar_percentage: 10,
                    building_area: 15000,
                    occupancy_rate: 0.8,
                    hvac_efficiency: 0.7,
                    month: new Date().getMonth() + 1,
                    season: Math.floor(new Date().getMonth() / 3) % 4
                })
            });
            const carbonData = await carbonResponse.json();
            if (carbonData.success) {
                updateCarbonDashboard(carbonData.prediction);
            }
        } catch (e) {
            console.log('Using fallback carbon data');
        }
        
        // Set interval for real-time updates
        setInterval(updateRealTime, 30000);
    }

    // Update Energy Dashboard KPIs and Charts
    function updateEnergyDashboard(prediction) {
        const predValue = prediction.prediction || prediction.current_emissions || 2500;
        
        // Update KPI card
        const kpiValue = document.querySelector('.kpi-card .kpi-value');
        if (kpiValue && kpiValue.textContent.includes('kWh')) {
            kpiValue.innerHTML = `${Math.round(predValue).toLocaleString()} <span>kWh</span>`;
        }
        
        // Update bar chart heights dynamically
        const bars = document.querySelectorAll('#page-energy .outcome-bar');
        const weekDays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
        const weekendFactor = [1, 1, 1, 1, 1, 0.6, 0.5];
        
        bars.forEach((bar, index) => {
            const height = 40 + Math.random() * 40 * weekendFactor[index];
            bar.style.height = height + '%';
            if (!bar.querySelector('span')) {
                const span = document.createElement('span');
                span.textContent = weekDays[index];
                bar.appendChild(span);
            }
        });
    }

    // Update Carbon Dashboard KPIs
    function updateCarbonDashboard(prediction) {
        const carbonValue = (prediction.current_emissions || 1200) / 1000;
        
        // Update carbon KPI
        const carbonKpi = document.querySelectorAll('.kpi-card');
        carbonKpi.forEach(card => {
            const label = card.querySelector('.kpi-label');
            if (label && label.textContent.includes('Carbon')) {
                const value = card.querySelector('.kpi-value');
                if (value) {
                    value.innerHTML = `${carbonValue.toFixed(2)} <span>tons</span>`;
                }
            }
        });
    }

    // ==================== FALLBACK CALCULATION FUNCTIONS ====================
    
    // Energy Prediction Calculation (fallback)
    function calculateEnergyPrediction(numBeds, equipHours, outdoorTemp, historicalAvg) {
        const basePerBed = 10;
        const equipFactor = equipHours / 24;
        let tempFactor = 1;
        if (outdoorTemp > 30 || outdoorTemp < 15) tempFactor = 1.3;
        else if (outdoorTemp > 25 || outdoorTemp < 18) tempFactor = 1.15;
        let predicted = (numBeds * basePerBed * equipFactor * tempFactor);
        return Math.round((predicted * 0.7) + (historicalAvg * 0.3));
    }

    // HVAC Optimization Calculation (fallback)
    function calculateHVACOptimization(currentTemp, occupancy, timeOfDay, weather) {
        let optimalTemp = 22;
        const occupancyMap = {'Low (0-20%)': 1, 'Medium (20-60%)': 0, 'High (60-100%)': -1};
        optimalTemp += occupancyMap[occupancy] || 0;
        const timeMap = {'Morning (6AM-12PM)': 0, 'Afternoon (12PM-6PM)': 1, 'Evening (6PM-10PM)': 0, 'Night (10PM-6AM)': -2};
        optimalTemp += timeMap[timeOfDay] || 0;
        const weatherMap = {'Sunny': -1, 'Cloudy': 0, 'Rainy': 1, 'Humid': -1};
        optimalTemp += weatherMap[weather] || 0;
        const tempDiff = Math.abs(currentTemp - optimalTemp);
        let savingsPercent = Math.min(tempDiff * 3, 25);
        if (currentTemp > optimalTemp) savingsPercent += 5;
        const monthlySavings = Math.round(50 * savingsPercent / 100 * 30 * 0.12);
        return { optimalTemp, savingsPercent: Math.round(savingsPercent * 10) / 10, monthlySavings };
    }

    // Carbon Emission Calculation (fallback)
    function calculateCarbonEmissions(energyConsumption, energySource) {
        const emissionFactors = {'Mixed Grid': 0.5, 'Grid + Solar': 0.35, 'Coal-based': 0.9, 'Renewable 100%': 0.05};
        const factor = emissionFactors[energySource] || 0.5;
        const dailyEmissions = (energyConsumption * factor) / 1000;
        return { dailyEmissions: Math.round(dailyEmissions * 100) / 100, targetEmissions: Math.round(dailyEmissions * 0.7 * 100) / 100 };
    }

    // Real-time Update Function
    function updateRealTime() {
        const currentPower = Math.round(120 + Math.random() * 80);
        const todayKwh = Math.round(3000 + Math.random() * 500);
        const peakPower = Math.round(150 + Math.random() * 40);
        const todayCost = Math.round(todayKwh * 0.12);
        
        const currentPowerEl = document.getElementById('currentPower');
        const todayKwhEl = document.getElementById('todayKwh');
        const peakPowerEl = document.getElementById('peakPower');
        
        if (currentPowerEl) currentPowerEl.textContent = currentPower + ' kW';
        if (todayKwhEl) todayKwhEl.textContent = todayKwh.toLocaleString();
        if (peakPowerEl) peakPowerEl.textContent = peakPower;
        
        const loadPercent = (currentPower / 200) * 100;
        const loadStatusEl = document.getElementById('loadStatus');
        const loadBarEl = document.getElementById('loadBar');
        const loadDescEl = document.getElementById('loadDesc');
        
        if (loadStatusEl && loadBarEl && loadDescEl) {
            loadBarEl.style.width = Math.min(loadPercent, 100) + '%';
            if (loadPercent > 85) { 
                loadStatusEl.className = 'risk-badge high'; 
                loadStatusEl.textContent = 'HIGH'; 
                loadDescEl.textContent = 'Power consumption exceeds safe limits!'; 
            }
            else if (loadPercent > 65) { 
                loadStatusEl.className = 'risk-badge medium'; 
                loadStatusEl.textContent = 'ELEVATED'; 
                loadDescEl.textContent = 'Power consumption elevated - monitor closely'; 
            }
            else { 
                loadStatusEl.className = 'risk-badge low'; 
                loadStatusEl.textContent = 'NORMAL'; 
                loadDescEl.textContent = 'Power consumption within normal range'; 
            }
        }
        // Push to realtime Chart.js series
        try {
            if (typeof realtimeChart !== 'undefined' && realtimeChart) {
                const labels = realtimeChart.data.labels || [];
                const ds = realtimeChart.data.datasets[0];
                const now = new Date();
                labels.push(now.toLocaleTimeString());
                ds.data.push(currentPower);
                // keep last 20 points
                if (labels.length > 20) { labels.shift(); ds.data.shift(); }
                realtimeChart.update();
            }
        } catch (chartErr) {
            console.warn('Realtime chart update failed', chartErr);
        }

        console.log('Real-time data updated');
    }

    // Department Analysis
    function calculateDepartmentAnalysis(dept, units) {
        const deptFactors = {'ICU': 45, 'Operating Rooms': 38, 'Laboratories': 28, 'Patient Wards': 18};
        const baseFactor = deptFactors[dept] || 20;
        const dailyConsumption = Math.round(units * baseFactor);
        const efficiency = Math.max(50, Math.min(95, 100 - (dailyConsumption / 50)));
        return { consumption: dailyConsumption, efficiency: Math.round(efficiency), monthly: dailyConsumption * 30 };
    }

    // Cost Analysis
    function calculateCostAnalysis(budget, spend) {
        const usagePercent = (spend / budget) * 100;
        return { usagePercent: Math.round(usagePercent * 10) / 10, remaining: budget - spend };
    }

    // Weather Impact
    function calculateWeatherImpact(temp, humidity) {
        let impactPercent = 0;
        if (temp > 25) impactPercent += (temp - 25) * 4;
        else if (temp < 18) impactPercent += (18 - temp) * 3;
        if (humidity > 60) impactPercent += (humidity - 60) * 0.3;
        
        let risk = 'low';
        if (impactPercent > 25) risk = 'high';
        else if (impactPercent > 15) risk = 'medium';
        
        return { impact: Math.round(impactPercent), risk };
    }

    // ==================== FORM HANDLERS ====================
    
    // Energy Prediction Form
    const energyForm = document.getElementById('energyForm');
    if (energyForm) {
        energyForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const numBeds = parseFloat(document.getElementById('numBeds').value) || 200;
            const equipHours = parseFloat(document.getElementById('equipHours').value) || 18;
            const outdoorTemp = parseFloat(document.getElementById('outdoorTemp').value) || 25;
            const historicalAvg = parseFloat(document.getElementById('historicalAvg').value) || 2200;
            
            // Call ML API
            fetch(`${API_BASE}/ml/predict/energy`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    num_beds: numBeds,
                    equipment_hours: equipHours,
                    outdoor_temp: outdoorTemp,
                    humidity: 60,
                    building_area: 15000,
                    day_of_week: new Date().getDay(),
                    is_weekend: (new Date().getDay() === 0 || new Date().getDay() === 6) ? 1 : 0,
                    occupancy_rate: 0.8,
                    hvac_efficiency: 0.75
                })
            })
            .then(response => response.json())
            .then(data => {
                let pred;
                if (data.success && data.prediction) {
                    pred = data.prediction.prediction;
                    console.log('✅ ML Energy prediction:', pred, 'kWh');
                } else {
                    throw new Error('API failed');
                }
                
                // Update main result
                const resultNumber = document.querySelector('#page-energy .result-card .result-number');
                if (resultNumber) resultNumber.textContent = Math.round(pred).toLocaleString();
                
                // Update analysis bars
                const lowEl = document.getElementById('energyLow');
                const highEl = document.getElementById('energyHigh');
                const avgEl = document.getElementById('energyAvg');
                const weeklyEl = document.getElementById('weeklyTotal');
                const dailyAvgEl = document.getElementById('dailyAvg');
                
                if (lowEl) lowEl.textContent = Math.round(pred * 0.85).toLocaleString();
                if (highEl) highEl.textContent = Math.round(pred * 1.15).toLocaleString();
                if (avgEl) avgEl.textContent = Math.round(pred).toLocaleString();
                if (weeklyEl) weeklyEl.textContent = Math.round(pred * 7).toLocaleString();
                if (dailyAvgEl) dailyAvgEl.textContent = Math.round(pred).toLocaleString();
                
                // Update weekly chart bars
                const bars = document.querySelectorAll('#page-energy .outcome-bar');
                const weekDays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
                const weekendFactor = [1, 1, 1, 1, 1, 0.6, 0.5];
                
                bars.forEach((bar, index) => {
                    const height = (pred / 3500) * 100 * weekendFactor[index];
                    bar.style.height = Math.min(height, 95) + '%';
                    const span = bar.querySelector('span');
                    if (span) span.textContent = weekDays[index];
                });
                
                // Update mini chart bars
                const miniBars = document.querySelectorAll('#page-energy .mini-chart .chart-bar');
                if (miniBars.length >= 3) {
                    miniBars[0].style.width = '60%';
                    miniBars[1].style.width = '75%';
                    miniBars[2].style.width = '90%';
                }
                
                // Update risk indicator
                const riskBadge = document.querySelector('#page-energy .risk-badge');
                const riskFill = document.querySelector('#page-energy .risk-fill');
                if (riskBadge && riskFill) {
                    if (pred > 3000) {
                        riskBadge.textContent = 'HIGH';
                        riskBadge.className = 'risk-badge high';
                        riskFill.style.width = '80%';
                    } else if (pred > 2500) {
                        riskBadge.textContent = 'MEDIUM';
                        riskBadge.className = 'risk-badge medium';
                        riskFill.style.width = '50%';
                    } else {
                        riskBadge.textContent = 'LOW';
                        riskBadge.className = 'risk-badge low';
                        riskFill.style.width = '30%';
                    }
                }
                // Update Chart.js charts for energy
                try {
                    if (typeof energyChart !== 'undefined' && energyChart) {
                        energyChart.data.datasets[0].data = [Math.round(pred * 0.85), Math.round(pred), Math.round(pred * 1.15)];
                        energyChart.update();
                    }
                    if (typeof energyTrendChart !== 'undefined' && energyTrendChart) {
                        const labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
                        const trend = labels.map((d,i) => {
                            const factor = (i >= 5) ? (0.65 + Math.random() * 0.2) : (0.9 + Math.random() * 0.2);
                            return Math.round(pred * factor / 7 * 7);
                        });
                        energyTrendChart.data.labels = labels;
                        energyTrendChart.data.datasets[0].data = trend;
                        energyTrendChart.update();
                    }
                } catch (chartErr) {
                    console.warn('Energy chart update failed', chartErr);
                }
            })
            .catch(() => {
                // Fallback to local calculation
                const predicted = calculateEnergyPrediction(numBeds, equipHours, outdoorTemp, historicalAvg);
                const resultNumber = document.querySelector('#page-energy .result-card .result-number');
                if (resultNumber) resultNumber.textContent = predicted.toLocaleString();
                
                if (document.getElementById('energyLow')) document.getElementById('energyLow').textContent = Math.round(predicted * 0.75).toLocaleString();
                if (document.getElementById('energyHigh')) document.getElementById('energyHigh').textContent = Math.round(predicted * 1.15).toLocaleString();
                if (document.getElementById('weeklyTotal')) document.getElementById('weeklyTotal').textContent = (predicted * 7).toLocaleString();
                
                console.log('📊 Local Energy prediction:', predicted, 'kWh');
            });
        });
    }

    // HVAC Optimization Form
    const hvacForm = document.getElementById('hvacForm');
    if (hvacForm) {
        hvacForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const currentTemp = parseFloat(document.getElementById('currentTemp').value) || 22;
            const occupancy = document.getElementById('occupancy')?.value || 'Medium (20-60%)';
            const timeOfDay = document.getElementById('timeOfDay')?.value || 'Afternoon (12PM-6PM)';
            const weather = document.getElementById('weather')?.value || 'Cloudy';
            
            // Call ML API
            fetch(`${API_BASE}/ml/predict/hvac`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    outdoor_temp: currentTemp + 5,
                    outdoor_humidity: 60,
                    indoor_temp_setpoint: currentTemp,
                    building_area: 15000,
                    num_occupants: 100,
                    equipment_load: 10000,
                    hvac_age_years: 5,
                    schedule_occupancy: 0.8,
                    weather_condition: 0,
                    current_efficiency: 0.65
                })
            })
            .then(response => response.json())
            .then(data => {
                let result;
                if (data.success && data.prediction) {
                    result = data.prediction;
                    console.log('✅ ML HVAC optimization:', result);
                } else {
                    throw new Error('API failed');
                }
                
                // Update temperature display
                const tempValue = document.querySelector('#page-hvac .temp-value');
                if (tempValue) tempValue.textContent = (result.optimal_setpoint || 22).toFixed(1);
                
                // Update savings
                const savingsItems = document.querySelectorAll('#page-hvac .savings-item .savings-value');
                if (savingsItems.length >= 2) {
                    savingsItems[0].textContent = (result.energy_savings_percent || 18).toFixed(1) + '%';
                    savingsItems[1].textContent = '$' + (result.projected_savings || 340);
                }
                
                // Update monthly savings
                const monthlySavingsEl = document.getElementById('hvacMonthlySavings');
                const energySavedEl = document.getElementById('hvacEnergySaved');
                if (monthlySavingsEl) monthlySavingsEl.textContent = '$' + (result.projected_savings || 340);
                if (energySavedEl) energySavedEl.textContent = Math.round((result.energy_savings_percent || 18) * 70);
                
                // Update outcome bars
                const hvacBars = document.querySelectorAll('#page-hvac .outcome-bar');
                const savingsData = [80, 72, 65, 58];
                hvacBars.forEach((bar, index) => {
                    bar.style.height = savingsData[index] + '%';
                });
                
                // Update mini chart
                const hvacMiniBars = document.querySelectorAll('#page-hvac .mini-chart .chart-bar');
                if (hvacMiniBars.length >= 3) {
                    hvacMiniBars[0].style.width = '85%';
                    hvacMiniBars[1].style.width = '55%';
                    hvacMiniBars[2].style.width = '35%';
                }
                // Update HVAC charts
                try {
                    if (typeof hvacChart !== 'undefined' && hvacChart) {
                        const currentEff = Math.round((result.current_efficiency || result.current_eff || 65));
                        const optimized = Math.round(result.optimal_setpoint ? result.optimal_setpoint : Math.max(40, currentEff - 10));
                        const potential = Math.max(0, currentEff - 20);
                        hvacChart.data.datasets[0].data = [currentEff, optimized, potential];
                        hvacChart.update();
                    }
                    if (typeof hvacTrendChart !== 'undefined' && hvacTrendChart) {
                        const weeks = ['W1','W2','W3','W4'];
                        const trend = weeks.map((w,i) => Math.max(30, Math.round((result.energy_savings_percent || 18) * (1 - i * 0.05) + 50)));
                        hvacTrendChart.data.labels = weeks;
                        hvacTrendChart.data.datasets[0].data = trend;
                        hvacTrendChart.update();
                    }
                } catch (chartErr) {
                    console.warn('HVAC chart update failed', chartErr);
                }
            })
            .catch(() => {
                // Fallback
                const result = calculateHVACOptimization(currentTemp, occupancy, timeOfDay, weather);
                const tempValue = document.querySelector('#page-hvac .temp-value');
                if (tempValue) tempValue.textContent = result.optimalTemp;
                
                const savingsItems = document.querySelectorAll('#page-hvac .savings-item .savings-value');
                if (savingsItems[0]) savingsItems[0].textContent = result.savingsPercent + '%';
                if (savingsItems[1]) savingsItems[1].textContent = '$' + result.monthlySavings;
                
                console.log('📊 Local HVAC optimization:', result);
            });
        });
    }

    // Carbon Forecast Form
    const carbonForm = document.getElementById('carbonForm');
    if (carbonForm) {
        carbonForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const energyConsumption = parseFloat(document.getElementById('energyConsumption').value) || 2500;
            const energySource = document.getElementById('energySource')?.value || 'Grid + Solar';
            
            // Call ML API
            fetch(`${API_BASE}/ml/predict/carbon`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    energy_consumption: energyConsumption * 24,
                    energy_source: energySource === 'Renewable 100%' ? 3 : (energySource === 'Coal-based' ? 2 : 0),
                    grid_mix_percentage: energySource.includes('Solar') ? 60 : 80,
                    solar_percentage: energySource.includes('Solar') ? 20 : 5,
                    building_area: 15000,
                    occupancy_rate: 0.8,
                    hvac_efficiency: 0.7,
                    month: new Date().getMonth() + 1,
                    season: Math.floor(new Date().getMonth() / 3) % 4
                })
            })
            .then(response => response.json())
            .then(data => {
                let result;
                if (data.success && data.prediction) {
                    result = data.prediction;
                    console.log('✅ ML Carbon forecast:', result);
                } else {
                    throw new Error('API failed');
                }
                
                // Update emission display
                const emissionValue = document.querySelector('#page-carbon .emission-value');
                if (emissionValue) emissionValue.textContent = ((result.current_emissions || 1200) / 1000).toFixed(2);
                
                // Update analysis values
                const carbonLow = document.getElementById('carbonLow');
                const carbonHigh = document.getElementById('carbonHigh');
                const carbonAvg = document.getElementById('carbonAvg');
                const carbonMonthly = document.getElementById('carbonMonthly');
                const carbonReduction = document.getElementById('carbonReduction');
                
                if (carbonLow) carbonLow.textContent = ((result.current_emissions || 1200) * 0.7 / 1000).toFixed(2);
                if (carbonHigh) carbonHigh.textContent = ((result.current_emissions || 1200) * 1.3 / 1000).toFixed(2);
                if (carbonAvg) carbonAvg.textContent = ((result.current_emissions || 1200) / 1000).toFixed(2);
                if (carbonMonthly) carbonMonthly.textContent = ((result.current_emissions || 1200) * 30 / 1000).toFixed(1);
                if (carbonReduction) carbonReduction.textContent = (result.reduction_potential || 30) + '%';
                
                // Update outcome bars
                const carbonBars = document.querySelectorAll('#page-carbon .outcome-bar');
                const carbonData = [75, 70, 65, 60];
                carbonBars.forEach((bar, index) => {
                    bar.style.height = carbonData[index] + '%';
                });
                
                // Update mini chart
                const carbonMiniBars = document.querySelectorAll('#page-carbon .mini-chart .chart-bar');
                if (carbonMiniBars.length >= 3) {
                    carbonMiniBars[0].style.width = '35%';
                    carbonMiniBars[1].style.width = '55%';
                    carbonMiniBars[2].style.width = '85%';
                }
                
                // Update comparison chart
                const compBars = document.querySelectorAll('#page-carbon .comparison-bar .bar-fill');
                if (compBars.length >= 2) {
                    compBars[0].style.width = '70%';
                    compBars[1].style.width = '50%';
                }
                // Update Carbon chart
                try {
                    const gridMix = (energySource && energySource.includes('Solar')) ? 60 : 80;
                    const solar = (energySource && energySource.includes('Solar')) ? 20 : 5;
                    const other = Math.max(0, 100 - gridMix - solar);
                    if (typeof carbonChart !== 'undefined' && carbonChart) {
                        carbonChart.data.datasets[0].data = [gridMix, solar, other];
                        carbonChart.update();
                    }
                } catch (chartErr) {
                    console.warn('Carbon chart update failed', chartErr);
                }
            })
            .catch(() => {
                // Fallback
                const result = calculateCarbonEmissions(energyConsumption, energySource);
                const emissionValue = document.querySelector('#page-carbon .emission-value');
                if (emissionValue) emissionValue.textContent = result.dailyEmissions;
                
                console.log('📊 Local Carbon forecast:', result);
            });
        });
    }

    // Department Form
    const deptForm = document.getElementById('deptForm');
    if (deptForm) {
        deptForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const dept = document.getElementById('deptSelect')?.value || 'ICU';
            const units = parseFloat(document.getElementById('deptUnits').value) || 20;
            
            const result = calculateDepartmentAnalysis(dept, units);
            
            if (document.getElementById('deptConsumption')) document.getElementById('deptConsumption').textContent = result.consumption;
            const ratingEl = document.getElementById('deptRating');
            if (ratingEl) {
                ratingEl.textContent = result.efficiency > 75 ? 'GOOD' : (result.efficiency > 60 ? 'FAIR' : 'POOR');
                ratingEl.className = 'risk-badge ' + (result.efficiency > 75 ? 'low' : (result.efficiency > 60 ? 'medium' : 'high'));
            }
            
            console.log('Department analysis:', result);
        });
    }

    // Cost Form
    const costForm = document.getElementById('costForm');
    if (costForm) {
        costForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const budget = parseFloat(document.getElementById('monthlyBudget').value) || 25000;
            const spend = parseFloat(document.getElementById('currentSpend').value) || 18200;
            
            const result = calculateCostAnalysis(budget, spend);
            
            if (document.getElementById('budgetUsage')) document.getElementById('budgetUsage').textContent = result.usagePercent;
            const statusEl = document.getElementById('budgetStatus');
            if (statusEl) {
                statusEl.textContent = result.usagePercent < 100 ? 'UNDER' : 'OVER';
                statusEl.className = 'risk-badge ' + (result.usagePercent < 80 ? 'low' : (result.usagePercent < 100 ? 'medium' : 'high'));
            }
            
            console.log('Cost analysis:', result);
        });
    }

    // Weather Form
    const weatherForm = document.getElementById('weatherForm');
    if (weatherForm) {
        weatherForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const temp = parseFloat(document.getElementById('weatherTemp').value) || 28;
            const humidity = parseFloat(document.getElementById('weatherHumidity').value) || 65;
            
            const result = calculateWeatherImpact(temp, humidity);
            
            if (document.getElementById('weatherImpact')) document.getElementById('weatherImpact').textContent = '+' + result.impact;
            const riskEl = document.getElementById('weatherRisk');
            if (riskEl) {
                riskEl.textContent = result.risk.toUpperCase();
                riskEl.className = 'risk-badge ' + result.risk;
            }
            
            console.log('Weather impact:', result);
        });
    }

    // Alert Form
    const alertForm = document.getElementById('alertForm');
    if (alertForm) {
        alertForm.addEventListener('submit', function(e) {
            e.preventDefault();
            alert('Alert thresholds saved successfully!');
        });
    }

    // Search
    const searchInput = document.querySelector('.search-box input');
    if (searchInput) {
        searchInput.addEventListener('input', function() { console.log('Searching:', this.value); });
    }

    // Download buttons
    document.querySelectorAll('.download-btn').forEach(btn => {
        btn.addEventListener('click', function() { alert('Report download started!'); });
    });

    // Save buttons
    document.querySelectorAll('.save-btn').forEach(btn => {
        btn.addEventListener('click', function() { alert('Settings saved successfully!'); });
    });

    console.log('MediVolt Dashboard initialized - API Base:', API_BASE);
});
