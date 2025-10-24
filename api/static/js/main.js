/**
 * Deepfake Detection Frontend JavaScript
 * Handles video upload, processing, and results display
 */

class DeepfakeDetector {
    constructor() {
        this.currentFile = null;
        this.currentResult = null;
        this.initializeEventListeners();
        this.initializeVideoPlayer();
    }

    initializeEventListeners() {
        // File upload
        const uploadInput = document.getElementById('video-upload');
        const uploadArea = document.getElementById('upload-area');
        const analyzeBtn = document.getElementById('analyze-btn');
        const removeFileBtn = document.getElementById('remove-file');
        const retryBtn = document.getElementById('retry-btn');

        uploadInput.addEventListener('change', this.handleFileUpload.bind(this));
        uploadArea.addEventListener('click', () => uploadInput.click());
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        analyzeBtn.addEventListener('click', this.analyzeVideo.bind(this));
        removeFileBtn.addEventListener('click', this.removeFile.bind(this));
        retryBtn.addEventListener('click', this.retryAnalysis.bind(this));

        // Video controls
        const showBboxes = document.getElementById('show-bboxes');
        const showHeatmaps = document.getElementById('show-heatmaps');

        showBboxes.addEventListener('change', this.updateVideoOverlay.bind(this));
        showHeatmaps.addEventListener('change', this.updateVideoOverlay.bind(this));
    }

    initializeVideoPlayer() {
        this.videoPlayer = document.getElementById('result-video');
        this.videoOverlay = document.getElementById('video-overlay');
    }

    handleFileUpload(event) {
        const file = event.target.files[0];
        if (file) {
            this.setCurrentFile(file);
        }
    }

    handleDragOver(event) {
        event.preventDefault();
        event.currentTarget.classList.add('border-blue-400', 'bg-blue-50');
    }

    handleDrop(event) {
        event.preventDefault();
        event.currentTarget.classList.remove('border-blue-400', 'bg-blue-50');

        const files = event.dataTransfer.files;
        if (files.length > 0) {
            this.setCurrentFile(files[0]);
        }
    }

    setCurrentFile(file) {
        // Validate file type
        const allowedTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/webm'];
        if (!allowedTypes.includes(file.type)) {
            this.showError('Please upload a valid video file (MP4, AVI, MOV, WebM)');
            return;
        }

        // Validate file size (100MB limit)
        const maxSize = 100 * 1024 * 1024; // 100MB
        if (file.size > maxSize) {
            this.showError('File size must be less than 100MB');
            return;
        }

        this.currentFile = file;
        this.displayFileInfo(file);
        this.enableAnalyzeButton();
    }

    displayFileInfo(file) {
        const fileName = document.getElementById('file-name');
        const fileSize = document.getElementById('file-size');
        const fileInfo = document.getElementById('file-info');

        fileName.textContent = file.name;
        fileSize.textContent = this.formatFileSize(file.size);
        fileInfo.classList.remove('hidden');
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    enableAnalyzeButton() {
        const analyzeBtn = document.getElementById('analyze-btn');
        analyzeBtn.disabled = false;
    }

    removeFile() {
        this.currentFile = null;
        document.getElementById('file-info').classList.add('hidden');
        document.getElementById('analyze-btn').disabled = true;
        document.getElementById('video-upload').value = '';
    }

    async analyzeVideo() {
        if (!this.currentFile) return;

        this.showProcessingSection();
        this.hideErrorSection();
        this.hideResultsSection();

        try {
            const formData = new FormData();
            formData.append('file', this.currentFile);
            formData.append('include_xai', 'true');

            const response = await fetch('/predict-with-annotations', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.currentResult = result;
            this.displayResults(result);

        } catch (error) {
            console.error('Analysis failed:', error);
            this.showError('Analysis failed: ' + error.message);
        } finally {
            this.hideProcessingSection();
        }
    }

    showProcessingSection() {
        document.getElementById('processing-section').classList.remove('hidden');
        this.updateProgress(0, 'Starting analysis...');
        this.updateProcessingSteps(['upload', 'extract', 'analyze', 'xai']);
    }

    hideProcessingSection() {
        document.getElementById('processing-section').classList.add('hidden');
    }

    updateProgress(percent, text) {
        const progressBar = document.getElementById('progress-bar');
        const progressText = document.getElementById('progress-text');
        const progressPercent = document.getElementById('progress-percent');

        progressBar.style.width = percent + '%';
        progressText.textContent = text;
        progressPercent.textContent = Math.round(percent) + '%';
    }

    updateProcessingSteps(activeSteps) {
        const steps = ['upload', 'extract', 'analyze', 'xai'];
        steps.forEach((step, index) => {
            const stepElement = document.getElementById(`step-${step}`);
            if (activeSteps.includes(step)) {
                stepElement.classList.remove('bg-gray-300');
                stepElement.classList.add('bg-blue-600');
            } else {
                stepElement.classList.remove('bg-blue-600');
                stepElement.classList.add('bg-gray-300');
            }
        });
    }

    displayResults(result) {
        this.showResultsSection();
        this.displayOverallResults(result);
        this.displayDetailedAnalysis(result);
        this.displayTimeline(result);
        this.displayHeatmaps(result);
    }

    showResultsSection() {
        document.getElementById('results-section').classList.remove('hidden');
    }

    hideResultsSection() {
        document.getElementById('results-section').classList.add('hidden');
    }

    displayOverallResults(result) {
        // Main result
        const resultIcon = document.getElementById('result-icon');
        const resultLabel = document.getElementById('result-label');
        const resultConfidence = document.getElementById('result-confidence');

        if (result.label === 'fake') {
            resultIcon.textContent = '🔴';
            resultLabel.textContent = 'FAKE';
            resultLabel.className = 'text-2xl font-bold mb-1 text-red-600';
        } else {
            resultIcon.textContent = '🟢';
            resultLabel.textContent = 'REAL';
            resultLabel.className = 'text-2xl font-bold mb-1 text-green-600';
        }

        resultConfidence.textContent = `Confidence: ${(result.score * 100).toFixed(1)}%`;

        // Statistics
        document.getElementById('processing-time').textContent = `${result.latency_ms}ms`;
        document.getElementById('frames-analyzed').textContent = result.meta.face_frames || 'N/A';
        document.getElementById('faces-detected').textContent = result.summary.total_faces_detected || 'N/A';

        // Summary
        const summaryStats = document.getElementById('summary-stats');
        summaryStats.innerHTML = `
            <div>Average Confidence: ${(result.summary.average_confidence * 100).toFixed(1)}%</div>
            <div>Fake Frames: ${result.summary.fake_frames || 0}</div>
            <div>Real Frames: ${result.summary.real_frames || 0}</div>
        `;
    }

    displayDetailedAnalysis(result) {
        // Overall explanation
        const overallExplanation = document.getElementById('overall-explanation');
        overallExplanation.textContent = result.explanations.overall_explanation || 'No explanation available';

        // Key findings
        const keyFindings = document.getElementById('key-findings');
        keyFindings.innerHTML = '';
        if (result.explanations.key_findings) {
            result.explanations.key_findings.forEach(finding => {
                const li = document.createElement('li');
                li.textContent = '• ' + finding;
                keyFindings.appendChild(li);
            });
        }

        // Confidence breakdown
        const confidenceBreakdown = document.getElementById('confidence-breakdown');
        if (result.explanations.confidence_factors) {
            const factors = result.explanations.confidence_factors;
            confidenceBreakdown.innerHTML = `
                <div>Max Confidence: ${(factors.max_confidence * 100).toFixed(1)}%</div>
                <div>Min Confidence: ${(factors.min_confidence * 100).toFixed(1)}%</div>
                <div>Standard Deviation: ${(factors.confidence_std * 100).toFixed(1)}%</div>
            `;
        }
    }

    displayTimeline(result) {
        const canvas = document.getElementById('timeline-chart');
        const ctx = canvas.getContext('2d');

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!result.frame_samples || result.frame_samples.length === 0) return;

        // Draw timeline
        const samples = result.frame_samples;
        const threshold = 0.5; // Default threshold

        // Draw background
        ctx.fillStyle = '#f3f4f6';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw threshold line
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, canvas.height * (1 - threshold));
        ctx.lineTo(canvas.width, canvas.height * (1 - threshold));
        ctx.stroke();

        // Draw confidence curve
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 2;
        ctx.beginPath();

        samples.forEach((sample, index) => {
            const x = (index / (samples.length - 1)) * canvas.width;
            const y = canvas.height * (1 - sample.score);

            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.stroke();

        // Draw points
        samples.forEach((sample, index) => {
            const x = (index / (samples.length - 1)) * canvas.width;
            const y = canvas.height * (1 - sample.score);

            ctx.fillStyle = sample.score >= threshold ? '#ef4444' : '#10b981';
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fill();
        });
    }

    displayHeatmaps(result) {
        const gallery = document.getElementById('heatmaps-gallery');
        gallery.innerHTML = '';

        if (!result.heatmaps || result.heatmaps.length === 0) {
            gallery.innerHTML = '<p class="text-gray-500">No heatmaps available</p>';
            return;
        }

        // Show first 8 heatmaps
        const heatmapsToShow = result.heatmaps.slice(0, 8);

        heatmapsToShow.forEach((heatmap, index) => {
            const img = document.createElement('img');
            img.src = heatmap;
            img.className = 'w-full h-32 object-cover rounded border';
            img.alt = `Heatmap ${index + 1}`;
            img.title = `Frame ${index + 1} - GRAD-CAM Heatmap`;
            gallery.appendChild(img);
        });
    }

    updateVideoOverlay() {
        // This would update the video overlay based on control settings
        // For now, it's a placeholder
        console.log('Video overlay updated');
    }

    showError(message) {
        const errorSection = document.getElementById('error-section');
        const errorMessage = document.getElementById('error-message');

        errorMessage.textContent = message;
        errorSection.classList.remove('hidden');
        this.hideProcessingSection();
    }

    hideErrorSection() {
        document.getElementById('error-section').classList.add('hidden');
    }

    retryAnalysis() {
        this.hideErrorSection();
        if (this.currentFile) {
            this.analyzeVideo();
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new DeepfakeDetector();
});
