// Common JavaScript functions for MedXpert

// File input preview
function readURL(input, previewElement) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        
        reader.onload = function(e) {
            $(previewElement).attr('src', e.target.result);
            $(previewElement).parent().removeClass('d-none');
        }
        
        reader.readAsDataURL(input.files[0]);
    }
}

// Show file name after selection
function updateFileLabel(input) {
    const fileName = input.files[0]?.name || "No file chosen";
    const fileLabel = $(input).siblings('.file-name');
    if (fileLabel.length) {
        fileLabel.text(fileName);
    }
}

// Format confidence percentage
function formatConfidence(confidence) {
    return parseFloat(confidence).toFixed(2) + '%';
}

// Show loading state
function showLoading(buttonElement) {
    $(buttonElement).prop('disabled', true);
    $(buttonElement).find('.spinner-border').removeClass('d-none');
}

// Hide loading state
function hideLoading(buttonElement) {
    $(buttonElement).prop('disabled', false);
    $(buttonElement).find('.spinner-border').addClass('d-none');
}

// Initialize Bootstrap 5 tooltips
$(function () {
    // Updated for Bootstrap 5
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });
});

// Enable all tooltips
function enableTooltips() {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });
}