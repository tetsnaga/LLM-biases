# LLM-biases
Cognitive Bias Quantification and Propagation in LLMs

Quick Review:<br>
    1. Run the requirements.txt file to install required depenedencies (pip install -r requirements.txt)<br>
    2. The Persona file (synthetic_climate_personas) and dataset (climate-fever-dataset.json) should be under Data/ folder.<br>
    3. All the results will go to outputs/ folder.<br>
    4. Template of the prompts are available in the prompts.txt file. 

To run the personExp01 file through command line, use following commands:<br>
    1. `chmod +x personaExp01.py`<br>
    2. `python personaExp01.py --model deepseek-r1:1.5b --personas Data/synthetic_climate_personas.csv --claims Data/climate-fever-dataset.csv`<br>
    (Model name, claim dataset path and person file path are required, optional inputs: temperature, number of personas and number of claims)