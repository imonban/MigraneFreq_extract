# A Large Language Model-Based Generative Nature Language Processing Framework Finetuned on Clinical Notes Accurately Extracts Headache Frequency from Clinic Notes

Headache frequency, defined as the number of days with any headache in a month (or four weeks), remains a key parameter in the evaluation of treatment response to migraine preventive medications. However, due to the variations and inconsistencies in documentation by clinicians, significant challenges exist to accurately extract headache frequency from the electronic health record (EHR) by traditional natural language processing (NLP) algorithms.

We design a generative GPT 2 transformer model using few-shot learning to extract the headache frequency.


To run the code: 

Step 1. Install the environment using yml file: conda env create -f environment_migraine.yml

Step 2. Download the model weights from here - https://drive.google.com/drive/folders/1DmGBddaUDX3DNHJAv132x9tIqkfqzKIG?usp=sharing

Step 3. Change infer.sh to provide the correct filepath and model weights, and also provide the correct path where you want to store the output

Step 4. Run the model ./infer.sh

The GPT-2 generative model was the best-performing model with an accuracy of 0.92 [0.91 – 0.93] and R2 score of 0.89 [0.87, 0.9], and all GPT2-based models outperformed the ClinicalBERT model in terms of the exact matching accuracy.

Creator: Imon Banerjee (Banerjee.Imon@mayo.edu), Man Luo (luo.man@mayo.edu)
