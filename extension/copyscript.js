function copyConsole() {
    const handler = async(event) => {
        if (!event.altKey) return;
        
        const selection = window.getSelection();
        const selectedText = selection.toString();
        
        if (!selectedText) {
            console.log('No text selected');
            return;
        }
        
        event.preventDefault();
        event.stopPropagation();
        console.log(selectedText);
    };
    document.addEventListener("keydown", handler, true);
    return () => document.removeEventListener("keydown", handler, true);
}

if (window._cleanup) window._cleanup();
window._cleanup = copyConsole();