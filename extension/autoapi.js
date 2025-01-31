function copyConsole() {
    const handler = async (event) => {
        if (!event.altKey) return;

        event.preventDefault();
        event.stopPropagation();

        for (let i = 0; i < 7; i++) {
            const cmdShiftRightEvent = new KeyboardEvent("keydown", {
                key: "ArrowLeft",
                code: "ArrowLeft",
                shiftKey: true, 
                metaKey: true,
                bubbles: true,
        });
        const shiftUpEvent = new KeyboardEvent("keydown", {
            key: "ArrowUp",
            code: "ArrowUp",
            shiftKey: true, 
            bubbles: true,
    });
            if (i == 0) {
                document.activeElement.dispatchEvent(cmdShiftRightEvent);
            } else {
                document.activeElement.dispatchEvent(shiftUpEvent);
            }

        }
        const selectedText = window.getSelection().toString();
        let gentext = " ";

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
                gentext = await response.json();
                console.log('Success:', gentext.data);
                gentext = gentext.data;
            } catch (error) {
                console.error('Error:', error);
            }
        };
        
        await sendPost();
        
        console.log(selectedText);

        const downArrowEvent = new KeyboardEvent("keydown", {
            key: "ArrowDown",
            code: "ArrowDown",
            bubbles: true,
        });
        document.activeElement.dispatchEvent(downArrowEvent);

        const editor = document.querySelector('.cm-editor');

        if (editor) {
            const view = editor.querySelector('.cm-content');
            if (view) {
                const selection = window.getSelection();
                const range = selection.getRangeAt(0);
                range.insertNode(document.createTextNode(gentext));
            }
        }
    };

    document.addEventListener("keydown", handler);
}



copyConsole();