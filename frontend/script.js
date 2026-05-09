const form = document.getElementById('assessmentForm');
const submitBtn = document.getElementById('submitBtn');
const btnText = document.getElementById('btnText');
const btnLoader = document.getElementById('btnLoader');
const errorDiv = document.getElementById('error');

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    errorDiv.textContent = '';
    submitBtn.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline-block';

    const formData = new FormData();
    formData.append('intraoral_image', document.getElementById('intraoralImage').files[0]);
    formData.append('histopath_image', document.getElementById('histopathImage').files[0]);
    formData.append('age', document.getElementById('age').value);
    formData.append('gender', document.getElementById('gender').value);
    formData.append('smoking', document.getElementById('smoking').value);
    formData.append('tobacco', document.getElementById('tobacco').value);
    formData.append('alcohol', document.getElementById('alcohol').value);
    formData.append('family_history', document.getElementById('familyHistory').value);
    formData.append('oral_lesions', document.getElementById('oralLesions').value);
    formData.append('unexplained_bleeding', document.getElementById('unexplainedBleeding').value);
    formData.append('difficulty_swallowing', document.getElementById('difficultySwallowing').value);
    formData.append('patches', document.getElementById('patches').value);

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Analysis failed');
        }

        localStorage.setItem('assessmentResult', JSON.stringify(data));
        window.location.href = 'result.html';
    } catch (error) {
        if (error.message === 'Failed to fetch') {
            errorDiv.textContent = 'Server not reachable. Make sure backend is running.';
        } else {
            errorDiv.textContent = error.message;
        }
        submitBtn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
});
