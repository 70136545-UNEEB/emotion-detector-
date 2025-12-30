// Emotion icons mapping
const emotionIcons = {
    'joy': '😊',
    'sadness': '😢', 
    'anger': '😠',
    'love': '❤️',
    'fear': '😨',
    'surprise': '😲'
};

// Emotion colors for tags
const emotionColors = {
    'joy': '#28a745',
    'sadness': '#6f42c1',
    'anger': '#dc3545',
    'love': '#e83e8c',
    'fear': '#ffc107',
    'surprise': '#17a2b8'
};

// Analyze single text
async function analyzeText() {
    console.log('🔍 Analyze text function called');
    
    const text = document.getElementById('textInput').value.trim();
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultsSection = document.getElementById('resultsSection');
    
    if (!text) {
        alert('⚠️ Please enter some text to analyze.');
        return;
    }
    
    // Show loading state
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    analyzeBtn.disabled = true;
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
            resultsSection.style.display = 'block';
        } else {
            alert('Error: ' + (data.error || 'Unknown error occurred'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to analyze text. Please try again.');
    } finally {
        // Reset button
        analyzeBtn.innerHTML = '<i class="fas fa-brain"></i> Analyze Emotion';
        analyzeBtn.disabled = false;
    }
}

// Analyze batch of texts
async function analyzeBatch() {
    console.log('📊 Analyze batch function called');
    
    const batchInput = document.getElementById('batchInput').value.trim();
    const batchBtn = document.getElementById('batchBtn');
    const batchResults = document.getElementById('batchResults');
    
    if (!batchInput) {
        alert('Please enter some texts to analyze.');
        return;
    }
    
    const texts = batchInput.split('\n').filter(text => text.trim());
    
    if (texts.length === 0) {
        alert('Please enter at least one non-empty text.');
        return;
    }
    
    // Show loading state
    batchBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    batchBtn.disabled = true;
    
    try {
        const response = await fetch('/analyze_batch', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ texts: texts })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayBatchResults(data);
            batchResults.style.display = 'block';
        } else {
            alert('Error: ' + (data.error || 'Unknown error occurred'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to analyze texts. Please try again.');
    } finally {
        // Reset button
        batchBtn.innerHTML = '<i class="fas fa-chart-bar"></i> Analyze Multiple Texts';
        batchBtn.disabled = false;
    }
}

function displayResults(data) {
    // Update emotion display
    document.getElementById('detectedEmotion').textContent = 
        data.emotion.charAt(0).toUpperCase() + data.emotion.slice(1);
    document.getElementById('detectedEmotion').className = `emotion-${data.emotion}`;
    
    // Update emotion icon
    document.getElementById('emotionIcon').textContent = emotionIcons[data.emotion] || '😐';
    document.getElementById('emotionIcon').className = `emotion-icon emotion-${data.emotion}`;
    
    // Update confidence
    document.getElementById('confidenceValue').textContent = 
        (data.confidence * 100).toFixed(1) + '%';
    
    // Update cleaned text
    document.getElementById('cleanedText').textContent = data.cleaned_text || 'Not available';
    
    // Update probability bars
    const probabilityBars = document.getElementById('probabilityBars');
    probabilityBars.innerHTML = '';
    
    if (data.probabilities) {
        Object.entries(data.probabilities)
            .sort(([,a], [,b]) => b - a)
            .forEach(([emotion, probability]) => {
                const barContainer = document.createElement('div');
                barContainer.className = 'probability-bar';
                
                const barLabel = document.createElement('div');
                barLabel.className = 'bar-label';
                barLabel.innerHTML = `
                    <span>${emotion.charAt(0).toUpperCase() + emotion.slice(1)}</span>
                    <span>${(probability * 100).toFixed(1)}%</span>
                `;
                
                const barFillContainer = document.createElement('div');
                barFillContainer.className = 'bar-container';
                
                const barFill = document.createElement('div');
                barFill.className = 'bar-fill';
                barFill.style.width = `${probability * 100}%`;
                barFill.style.backgroundColor = emotionColors[emotion] || '#6c757d';
                
                barFillContainer.appendChild(barFill);
                barContainer.appendChild(barLabel);
                barContainer.appendChild(barFillContainer);
                probabilityBars.appendChild(barContainer);
            });
    }
}

function displayBatchResults(data) {
    const batchResultsContent = document.getElementById('batchResultsContent');
    batchResultsContent.innerHTML = '';
    
    if (data.results && data.results.length > 0) {
        data.results.forEach((result, index) => {
            const batchItem = document.createElement('div');
            batchItem.className = 'batch-item';
            
            batchItem.innerHTML = `
                <div class="text-content">
                    <strong>${index + 1}.</strong> ${result.text}
                </div>
                <div class="emotion-result">
                    <span class="emotion-tag" style="background-color: ${emotionColors[result.emotion] || '#6c757d'}">
                        ${emotionIcons[result.emotion] || '😐'} ${result.emotion.toUpperCase()}
                    </span>
                    <span class="confidence-badge">${(result.confidence * 100).toFixed(1)}%</span>
                </div>
            `;
            
            batchResultsContent.appendChild(batchItem);
        });
    } else {
        batchResultsContent.innerHTML = '<p>No results to display.</p>';
    }
}

function addSampleText() {
    const sampleTexts = [
        "I am feeling absolutely wonderful today!",
        "This makes me so angry I can't stand it.",
        "I'm scared about what might happen tomorrow.",
        "I love this so much, it's amazing!",
        "I feel so sad and lonely right now.",
        "Wow, that was completely unexpected!"
    ];
    
    const randomText = sampleTexts[Math.floor(Math.random() * sampleTexts.length)];
    document.getElementById('textInput').value = randomText;
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('Emotion Detector loaded!');
    
    // Make functions globally available
    window.analyzeText = analyzeText;
    window.analyzeBatch = analyzeBatch;
    window.addSampleText = addSampleText;
    window.displayResults = displayResults;
    window.displayBatchResults = displayBatchResults;
});