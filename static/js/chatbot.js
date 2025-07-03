document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const conversation = document.getElementById('conversation');

    // --- File Upload Logic ---
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!fileInput.files.length) {
            uploadStatus.innerHTML = `<div class="alert alert-warning">Please select a file to upload.</div>`;
            return;
        }

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        uploadStatus.innerHTML = `<div class="alert alert-info">Uploading and processing...</div>`;

        try {
            const response = await fetch('/api/chatbot/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                uploadStatus.innerHTML = `<div class="alert alert-success">${result.message}</div>`;
                // TODO: Refresh document list in a later phase
            } else {
                throw new Error(result.message || 'Upload failed');
            }
        } catch (error) {
            uploadStatus.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
        } finally {
            uploadForm.reset();
        }
    });

    // --- Chat Logic ---
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = messageInput.value.trim();
        if (!question) return;

        // Display user's message
        addMessageToChat('user', question);
        messageInput.value = '';

        // Show typing indicator
        showTypingIndicator();

        try {
            const response = await fetch('/api/chatbot/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question })
            });

            const result = await response.json();
            hideTypingIndicator();

            if (response.ok) {
                addMessageToChat('assistant', result.answer);
            } else {
                throw new Error(result.message || 'Failed to get a response');
            }
        } catch (error) {
            hideTypingIndicator();
            addMessageToChat('assistant', `Sorry, an error occurred: ${error.message}`);
        }
    });

    function addMessageToChat(sender, text) {
        const bubble = document.createElement('div');
        bubble.classList.add('chat-bubble', sender);
        bubble.textContent = text;
        conversation.appendChild(bubble);
        conversation.scrollTop = conversation.scrollHeight; // Auto-scroll to the bottom
    }
    
    let typingIndicator;
    function showTypingIndicator() {
        typingIndicator = document.createElement('div');
        typingIndicator.classList.add('chat-bubble', 'assistant', 'typing-indicator');
        typingIndicator.innerHTML = `<span></span><span></span><span></span>`;
        conversation.appendChild(typingIndicator);
        conversation.scrollTop = conversation.scrollHeight;
    }

    function hideTypingIndicator() {
        if (typingIndicator) {
            conversation.removeChild(typingIndicator);
        }
    }
});