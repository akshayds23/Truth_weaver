async function analyze(event) {
  event.preventDefault();

  const subjectId = document.getElementById('subjectId').value.trim();
  const filesInput = document.getElementById('files');
  const saveOutputs = document.getElementById('saveOutputs').checked;
  const whisperModel = document.getElementById('whisperModel').value;
  const statusEl = document.getElementById('status');
  const resultEl = document.getElementById('result');

  if (!subjectId) {
    alert('Subject ID is required');
    return;
  }
  if (!filesInput.files.length) {
    alert('Please select at least one audio/text file');
    return;
  }

  const formData = new FormData();
  formData.append('subject_id', subjectId);
  formData.append('whisper_model', whisperModel);
  formData.append('save_outputs', String(saveOutputs));
  for (const file of filesInput.files) {
    formData.append('files', file, file.name);
  }

  statusEl.textContent = 'Uploading and analyzing... this may take a while for long audio.';
  resultEl.textContent = '';

  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status}: ${text}`);
    }
    const json = await res.json();
    resultEl.textContent = JSON.stringify(json, null, 2);
    statusEl.textContent = 'Done.';
  } catch (err) {
    console.error(err);
    statusEl.textContent = 'Error: ' + err.message;
  }
}

document.getElementById('analyze-form').addEventListener('submit', analyze);
