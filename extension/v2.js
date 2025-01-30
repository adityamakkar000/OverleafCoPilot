function copyConsole() {
    const handler = async (event) => {
        if (!event.altKey) return;
        event.preventDefault();
        event.stopPropagation();

        // Try to get the editor from Overleaf's global object
        const editor = document.querySelector('.cm-editor');
        console.log("Found editor:", editor); // Debug log

        if (editor) {
            const view = editor.querySelector('.cm-content');
            if (view) {
                const text = 'himynameis';
                // Insert at current selection
                const selection = window.getSelection();
                const range = selection.getRangeAt(0);
                range.insertNode(document.createTextNode(text));
            }
        }
    };

    document.addEventListener("keydown", handler, true);
    return () => document.removeEventListener("keydown", handler, true);
}

if (window._cleanup) window._cleanup();
window._cleanup = copyConsole();