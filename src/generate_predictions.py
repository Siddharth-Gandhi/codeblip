import os
import random
import warnings

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import logging
import pickle

from codeblip_qformer import CodeQformer
from dataset import CodeTranslationDataset
from codeblip_llama import Blip2Llama

logging.set_verbosity_error()  # Only show errors, not warnings

source_lang = 'java'
target_lang = 'cs'
batch_size = 1


if __name__ == "__main__":
    stage1_checkpoint = 'models/stage1_out/stage1_best.pt'
    stage1_model = CodeQformer.from_config({'pretrained_path': stage1_checkpoint})

    stage1_qformer = stage1_model.Qformer
    stage1_query_tokens = stage1_model.query_tokens

    model = Blip2Llama(stage1_qformer, stage1_query_tokens).to('cuda' if torch.cuda.is_available() else 'cpu')

    stage_2_checkpoint = 'models/stage2_out_32/llama_stage2_best.pt'
    model.load_state_dict(torch.load(stage_2_checkpoint))
    model.eval()

    cs_test_dir = "/data/tir/projects/tir6/general/piyushkh/anlp/codeblip/data/exps"

    for filename in tqdm(os.listdir(cs_test_dir)):
        if filename.endswith(f".{source_lang}"):
            
            test_source_file, test_target_file = os.path.join(cs_test_dir, filename.split('.')[0]+f".{source_lang}"), os.path.join(cs_test_dir, filename.split('.')[0]+f".{target_lang}")

            try:
                test_dataset = CodeTranslationDataset(test_source_file, test_target_file)
            except:
                print("File not found source: ", test_source_file)
                print("File not found target: ", test_target_file)
                continue
            # test_loader = DataLoader(test_dataset, batch_size=batch_size)

            model.eval()
            predictions = []
            error_count=0
            error_samples=[]
            with torch.no_grad():
                for batch in tqdm(test_dataset, desc="Testing"):
                    try:
                        predictions.append(model.generate(batch))
                    except:
                        error_count+=1
                        error_samples.append(batch)
                        predictions.append("")
                        continue
            print("Error count", error_count)

            #Save predictions in a txt file
            save_file_name = filename.split('.')[0] + 'predictions_cs.txt'
            with open(f'/data/tir/projects/tir6/general/piyushkh/anlp/codeblip/data/exps_results/{save_file_name}', 'w') as f:
                for item in predictions:
                    if item=="":
                        f.write("%s\n" % "     ")
                    f.write("%s\n" % item)


    #save error samples
    # with open('error_samples_512.pkl', 'wb') as f:
    #     pickle.dump(error_samples, f)
    
    # #Save prediction in a pkl file
    # with open('codeblip_predictions_512.pkl', 'wb') as f:
    #     pickle.dump(predictions, f)


#Read predictions from pkl file
# with open('codeblip_predictions.pkl', 'rb') as f:
#     predictions = pickle.load(f)



# with open('codeblip_predictions_512.txt', 'w') as f:
#     for item in predictions:
#         if item=="":
#             f.write("%s\n" % "                         ")
#         else:
#             f.write("%s\n" % item)