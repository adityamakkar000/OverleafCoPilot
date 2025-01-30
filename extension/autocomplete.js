function copyConsole() {
    const handler = async (event) => {
        if (!event.altKey) return;

        event.preventDefault();
        event.stopPropagation();

        for (let i = 0; i < 6; i++) {
            const shiftUpEvent = new KeyboardEvent("keydown", {
                key: "ArrowUp",
                code: "ArrowUp",
                shiftKey: true, 
                bubbles: true,
            });

            const focusedElement = document.activeElement;
            if (focusedElement) {
                focusedElement.dispatchEvent(shiftUpEvent);
            } else {
                document.dispatchEvent(shiftUpEvent);
            }
        }

        const selection = window.getSelection();
        const selectedText = selection.toString();
        
        console.log(selectedText);

        const downArrowEvent = new KeyboardEvent("keydown", {
            key: "ArrowDown",
            code: "ArrowDown",
            bubbles: true,
        });

        const focusedElement = document.activeElement;
        if (focusedElement) {
            focusedElement.dispatchEvent(downArrowEvent);
        } else {
            document.dispatchEvent(downArrowEvent);
        }

        const editor = document.querySelector('.cm-editor');
        //console.log("Found editor:", editor); 

        if (editor) {
            const view = editor.querySelector('.cm-content');
            if (view) {
                const text = 'himynameis';
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