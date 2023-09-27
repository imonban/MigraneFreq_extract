import argparse
from tqdm import tqdm
import math
import os
import pandas as pd
import sys
from infer import process
from extract import sentence_extraction

def main():
    
    # Create the parser
    parser = argparse.ArgumentParser(description='This is a script for extracting migrane frequency from clinic notes.')

    # Add the arguments
    parser.add_argument('--model_name', type=str, default="microsoft/BioGPT-Large",
                    help='the name of the evaluated model')
    parser.add_argument('--device',  type=str, default="0",
                    help='device id')
    parser.add_argument('--dataset_name',  type=str, default="Question-Answering_BioASQ-8b_yesno_classification",
                    help='provide a .xlsx file where CLINICAL_DOCUMENT_TEXT column stores the note text.')
    parser.add_argument('--output_file',  type=str, required=True,
                    help='the output prediction path')
    parser.add_argument('--batch_size',  type=int, default=1,
                    help='batch size of the testing time')

    # Parse the arguments
    args = parser.parse_args()
    model_name = args.model_name
    batch_size = args.batch_size
    device = "cuda:{}".format(args.device)
    output_file = args.output_file
    #try:
    dataset = pd.read_excel(args.dataset_name)
    print("total inference:", len(dataset))
    #except:
    #    print('Can not read the file; please provide a correct csv path')
    #    sys.exit()
    try:
        dataset['SENT'] = dataset['CLINICAL_DOCUMENT_TEXT'].apply(sentence_extraction)
        process(model_name, dataset, batch_size,device, output_file)
    except:
        print('Issues in running: exiting!')
        sys.exit()
    
if __name__ == "__main__":
    main()