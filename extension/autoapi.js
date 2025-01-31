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

            document.activeElement.dispatchEvent(shiftUpEvent);
        }
        const selectedText = window.getSelection().toString();
        const sendPost = async () => {
            try {
                const url = 'http://10.39.26.240:3000/post'; 
                const body = JSON.stringify({ text: selectedText });
                const headers = { 'Content-Type': 'application/json'};
                const method = 'POST';
                const response = await fetch(url, { 
                    method, 
                    headers, 
                    body,
                });
                const data = await response.text();
                console.log('Success:', data);
            } catch (error) {
                console.error('Error:', error);
            }
        };
        
        sendPost();
        
        console.log(selectedText);

        const downArrowEvent = new KeyboardEvent("keydown", {
            key: "ArrowDown",
            code: "ArrowDown",
            bubbles: true,
        });

        document.activeElement.dispatchEvent(downArrowEvent);

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

    document.addEventListener("keydown", handler);
}

copyConsole();