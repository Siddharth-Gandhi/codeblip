from transformers import AutoTokenizer
import transformers
import torch
import sys
from tqdm import tqdm
from dataset import CodeTranslationDataset

# /data/datasets/models/huggingface/meta-llama/CodeLlama-34b-Instruct-hf
# /data/datasets/models/huggingface/meta-llama/CodeLlama-13b-Instruct-hf

source_lang = 'java'
target_lang = 'cs'
model_path = "/data/datasets/models/huggingface/meta-llama/CodeLlama-7b-Instruct-hf"

class InstructLlama:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            # torch_dtype=torch.float32, # for CPU
            torch_dtype=torch.float16, # for GPU
            device_map="auto",
        )
    def generate_code(self, prompt):
        sequences = self.pipeline(
            prompt,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=100,
            return_full_text=False,
            top_p=0.9,
        )
        result = sequences[0]   
        return result

def hf_generate_code(model, prompt):
    # print("Input Prompt: ", prompt)
    result = model.generate_code(prompt)
    print("Final Result: ", result)
    return result


if __name__ == "__main__":
    test_source_file, test_target_file = f'data/test.java-cs.txt.{source_lang}', f'data/test.java-cs.txt.{target_lang}'
    test_dataset = CodeTranslationDataset(test_source_file, test_target_file, is_llama=True)
    model = InstructLlama(model_path)
    predictions = []
    i=0
    for sample in tqdm(test_dataset):
        java_code = sample
        
        prompt = f"""You are an expert at translating code from one language to another.
        Translate the provided code in Java to C#. """ + """Use the following format for translation:
        Java: public class HelloWorld { public static void main(String[] args) { System.out.println(\"Hello, world!\"); } } 
        C#: public class HelloWorld { public static void Main(string[] args) { System.Console.WriteLine(\"Hello, world!\"); } }
        
        Java: public class Test { public static int add(int a, int b) { return a + b; } }
        C#: public class Test { public static int Add(int a, int b) { return a + b; } }

        Java: public String toString() {return getClass().getName() + " [" +_value +"]";}
        C#: public override String ToString(){StringBuilder sb = new StringBuilder(64);sb.Append(GetType().Name).Append(" [");sb.Append(value);sb.Append("]");return sb.ToString();}
        """ + f" Java: {java_code} \n C#:"

        predictions.append(hf_generate_code(model, prompt))
        i+=1
        if i==3:
            break
    print("Predictions", predictions)