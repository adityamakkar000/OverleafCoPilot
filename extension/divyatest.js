function copyConsole() {
    console.log('Setting up copy handler...'); 
    
    const handler = async(event) => {
        console.log('Key event:', {
            key: event.key,
            altKey: event.altKey,
            code: event.code,
            which: event.which
        });
        
        if (!event.altKey) return;
        
        const selection = window.getSelection();
        const selectedText = selection.toString();
        
        if (!selectedText) {
            console.log('No text selected');
            return;
        }
        
        event.preventDefault();
        event.stopPropagation();
        
        try {
            await navigator.clipboard.writeText(selectedText);
            console.log('Successfully copied text:', selectedText);
        } catch(e) {
            console.error('Clipboard API failed:', e);
            
            try {
                const textarea = document.createElement('textarea');
                textarea.value = selectedText;
                textarea.style.position = 'fixed';
                textarea.style.opacity = '0';
                document.body.appendChild(textarea);
                textarea.select();
                
                const success = document.execCommand('copy');
                console.log('Fallback copy result:', success);
                document.body.removeChild(textarea);
            } catch(e2) {
                console.error('Fallback method failed:', e2);
            }
        }
    };

}

// Clear any existing handlers and start fresh
if (window._cleanup) window._cleanup();
window._cleanup = copyConsole();