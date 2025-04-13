import torch
import json, pickle, argparse, os
import multiprocessing as mp
from synllama.llm.vars import *
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

instruction = TEMPLATE["instruction"]
input_template = TEMPLATE["input"]

def generate_text(smiles, tokenizer, model, stopping_ids, sampling_params, max_length=1600):
    input = input_template.replace("SMILES_STRING", smiles)
    prompt_complete = "### Instruction:\n" + instruction + "\n\n### Input:\n"+ input + "\n\n### Response: \n"
    inputs = tokenizer(prompt_complete, return_tensors="pt").to(model.device)
    prompt_length = inputs.input_ids.shape[1]
    
    generated_texts = []
    
    for params in sampling_params:
        temp = params["temp"]
        top_p = params["top_p"]
        repeat = params["repeat"]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=temp,
                top_p=top_p,
                num_return_sequences=repeat,
                eos_token_id=stopping_ids,
                pad_token_id=tokenizer.eos_token_id
            )
        for output in outputs:
            generated_text = tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
            generated_texts.append(generated_text.strip())
    
    return generated_texts

def process_batch(args):
    gpu_id, model_path, smiles_batch, sampling_params = args
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if 'cuda' in device else torch.float32,
        device_map={'': device}
    )
    
    stopping_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    
    results = {}
    for smiles in tqdm(smiles_batch, desc=f"Processing on {device.upper()}"):
        try:
            response = generate_text(smiles, tokenizer, model, stopping_ids, sampling_params)
            json_responses = []
            for r in response:
                try:
                    json_responses.append(json.loads(r))
                except json.JSONDecodeError:
                    json_responses.append("json format error")
            results[smiles] = json_responses
        except Exception as e:
            results[smiles] = f"Error: {str(e)}"
    
    return results

def main(model_path, smiles_path, save_path, sampling_params, gpus = None):
    with open(smiles_path, "r") as f:
        smiles_list = [line.strip() for line in f]
    
    num_gpus = torch.cuda.device_count() if gpus is None else gpus
    print(f"Number of available GPUs: {num_gpus}")
    
    if num_gpus > 1:
        pool = mp.Pool(num_gpus)
        try:
            # Process batches on different GPUs
            batches = [smiles_list[i::num_gpus] for i in range(num_gpus)]
            results = pool.map(process_batch, [(i, model_path, batch, sampling_params) for i, batch in enumerate(batches)])
            
            # Combine results from all GPUs
            combined_results = {}
            for r in results:
                combined_results.update(r)
                
        finally:
            # Ensure pool cleanup happens even if an error occurs
            pool.close()  # Prevent any more tasks from being submitted
            pool.join()   # Wait for all processes to finish
            pool.terminate()  # Terminate all worker processes
    else:
        # If only one GPU, process all SMILES on that GPU
        combined_results = process_batch((0, model_path, smiles_list, sampling_params))
    
    # Save results
    with open(save_path, "wb") as f:
        pickle.dump(combined_results, f)

    # close pool
    return combined_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference pipeline for reaction prediction")
    parser.add_argument("--model_path", type=str, help="Path to the model", default="data/model/SynLlama-1B-2M")
    parser.add_argument("--smiles_path", type=str, help="Path to the SMILES file")
    parser.add_argument("--save_path", type=str, help="Pickle file path to save the results", default = None)
    parser.add_argument("--sample_mode", type=str, default=None, help="Sampling mode, choose from: greedy, frugal, frozen_only, low_only, medium_only, high_only")
    parser.add_argument("--temp", type=float, default=None, help="Temperature for the model")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p for the model")
    parser.add_argument("--repeat", type=int, default=None, help="Number of times to repeat the model")
    parser.add_argument("--gpus", type=int, default=None, help="name of the cuda device to use, default is all available GPUs")
    args = parser.parse_args()
    mp.set_start_method('spawn', force=True)
    if args.save_path is None:
        args.save_path = args.smiles_path.replace(".smi", "_results.pkl")
    directory = os.path.dirname(args.save_path)
    os.makedirs(directory, exist_ok=True)
    sample_mode_mapping = {
        "greedy": sampling_params_greedy,
        "frugal": sampling_params_frugal,
        "frozen_only": sampling_params_frozen_only,
        "low_only": sampling_params_low_only,
        "medium_only": sampling_params_medium_only,
        "high_only": sampling_params_high_only
    }
    if args.sample_mode is None:
        assert args.temp is not None and args.top_p is not None and args.repeat is not None, "Please provide a sample mode or all the sampling parameters"
        sampling_params = [
            {"temp": args.temp, "top_p": args.top_p, "repeat": args.repeat}
        ]
    else:
        assert args.sample_mode in sample_mode_mapping, f"Invalid sample mode: {args.sample_mode}"
        sampling_params = sample_mode_mapping[args.sample_mode]

    main(args.model_path, args.smiles_path, args.save_path, sampling_params=sampling_params, gpus=args.gpus)