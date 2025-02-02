![](https://github.com/adityamakkar000/OverleafCoPilot/blob/main/public/image.png)

## Github Copilot but for Overleaf. 

A Chrome extension and finetuning script to run a local LLM latex completion. 


## Getting Started

1. **Clone the Repository**

   ```bash
   git clone https://github.com/adityamakkar000/ASL-Sign-Research.git
   ```
3. **Setup Env**

    Setup the python environment using
    ```bash
   pip install -r req.txt
    ```
    
4. **Finetune**

    Place your latex ```.txt``` in  ```ai/dataset/``` and create a .yaml file in ```ai/configs/``` following the structure in ```ai/configs/main.yaml```
    Run
    ```bash
    python3 finetune.py --config-name $config-name arg1=$arg1 arg2=$arg2 ...
    ```
   
4. **Use**

   Run a local server using 
    ```bash
    python3 main.py path=$(path to finetuned model) tokens=$tokens
     ```
    Add the ip-address in ```extension/sentence.js``` and  load the extension folder in ```chrome://extensions/```
    Add ip-address at ```chrome://flags/``` to allow for mixed-content on http and https
    Reload chrome and press ```alt``` to run inference 
  
