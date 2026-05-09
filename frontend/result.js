const data = JSON.parse(localStorage.getItem('assessmentResult'));

if (!data) {
    window.location.href = 'index.html';
} else {
    // Display final score and risk
    if (data.final_score !== undefined) {
        document.getElementById('finalScore').textContent = data.final_score.toFixed(1) + '%';
        const finalRiskEl = document.getElementById('finalRiskLabel');
        finalRiskEl.textContent = data.final_risk;
        finalRiskEl.className = 'risk-label ' + data.final_risk.toLowerCase();
    }
    
    // Display breakdown scores
    if (data.intraoral_score !== undefined) {
        const intraoralText = data.intraoral_score === 0 ? 'N/A' : data.intraoral_score.toFixed(1) + '%';
        document.getElementById('intraoralScore').textContent = intraoralText;
        document.getElementById('intraoralProgress').style.width = data.intraoral_score + '%';
    }
    
    if (data.histopath_score !== undefined) {
        const histopathText = data.histopath_score === 0 ? 'N/A' : data.histopath_score.toFixed(1) + '%';
        document.getElementById('histopathScore').textContent = histopathText;
        document.getElementById('histopathProgress').style.width = data.histopath_score + '%';
    }
    
    document.getElementById('clinicalScore').textContent = data.clinical_score.toFixed(1) + '%';
    document.getElementById('clinicalProgress').style.width = data.clinical_score + '%';
    
    document.getElementById('recommendation').textContent = data.recommendation;
    document.getElementById('summary').textContent = data.summary;
    document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(0) + '%';
    
    // Display heatmap if available
    if (data.heatmap_url) {
        document.getElementById('heatmapCard').style.display = 'block';
        document.getElementById('heatmapImage').src = 'http://127.0.0.1:5000' + data.heatmap_url;
    }
    
    // Display feature importance
    const featureDiv = document.getElementById('featureImportance');
    if (data.feature_importance && Object.keys(data.feature_importance).length > 0) {
        for (const [key, value] of Object.entries(data.feature_importance)) {
            const item = document.createElement('div');
            item.className = 'feature-item';
            item.innerHTML = `<span>${key}</span><span>${value}%</span>`;
            featureDiv.appendChild(item);
        }
    } else {
        featureDiv.innerHTML = '<div class="feature-item"><span>No significant risk factors detected</span></div>';
    }
}

// Doctor Feedback Logic
let selectedDecision = null;

document.getElementById('acceptBtn').addEventListener('click', () => {
    selectedDecision = 'accept';
    document.getElementById('reasonSection').style.display = 'none';
    highlightButton('acceptBtn');
});

document.getElementById('reviewBtn').addEventListener('click', () => {
    selectedDecision = 'review';
    document.getElementById('reasonSection').style.display = 'block';
    highlightButton('reviewBtn');
});

document.getElementById('rejectBtn').addEventListener('click', () => {
    selectedDecision = 'reject';
    document.getElementById('reasonSection').style.display = 'block';
    highlightButton('rejectBtn');
});

function highlightButton(btnId) {
    document.querySelectorAll('.feedback-btn').forEach(btn => btn.style.opacity = '0.5');
    document.getElementById(btnId).style.opacity = '1';
}

document.getElementById('submitFeedback').addEventListener('click', async () => {
    if (!selectedDecision) {
        alert('Please select a decision');
        return;
    }

    const reason = document.getElementById('feedbackReason').value;
    const notes = document.getElementById('feedbackNotes').value;

    if ((selectedDecision === 'review' || selectedDecision === 'reject') && !reason) {
        alert('Please select a reason');
        return;
    }

    const feedbackData = {
        decision: selectedDecision,
        reason: reason || null,
        notes: notes || null,
        result: data
    };

    try {
        const response = await fetch('http://127.0.0.1:5000/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(feedbackData)
        });

        if (response.ok) {
            document.getElementById('submitFeedback').disabled = true;
            document.getElementById('feedbackMessage').textContent = 'Feedback submitted successfully';
            document.getElementById('feedbackMessage').style.display = 'block';
        } else {
            alert('Failed to submit feedback');
        }
    } catch (error) {
        alert('Error submitting feedback: ' + error.message);
    }
});
