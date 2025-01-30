function copyConsole() {
    const handler = async (event) => {

        // console.log(event);
        if (!event.altKey) return;

        event.preventDefault();
        event.stopPropagation();

        const activeLine = document.querySelector('.cm-activeLine');
        if (!activeLine) {
            console.log("No active line found");
            return;
        } else {
            console.log("Active line content:", activeLine.textContent);
        }
        // If the element is contenteditable or an input/textarea, we can also set its value directly
            const currentValue = activeLine.value || activeLine.textContent;
            const newValue = currentValue + 'himynameis';
            if (activeLine.value !== undefined) {
                activeLine.value = newValue;
            } else {
                activeLine.textContent = newValue;
            }
            console.log("currentvalue:", currentValue);
        
        console.log("Attempted to insert 'a' into:", activeLine);

    };

    document.addEventListener("keydown", handler, true);
    return () => document.removeEventListener("keydown", handler, true);
}

if (window._cleanup) window._cleanup();
window._cleanup = copyConsole();