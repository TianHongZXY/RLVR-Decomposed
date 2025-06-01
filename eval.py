import os
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from utils import extract_answer_math
from grader import math_equal
os.environ["NCCL_DEBUG"] = "WARN"


def prepare_data(example, prompt_key):
    qwen_boxed_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n"
    example['prompt'] = qwen_boxed_prompt.replace("{input}", example[prompt_key])

    return example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--datasets", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--num_generation", type=int, default=1)
    parser.add_argument("--dataset_num_proc", type=int, default=1)
    parser.add_argument("--resume_id", type=int, default=0)
    parser.add_argument("--comment", type=str, default="")
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.model_name):
        print(f"Model {args.model_name} not found. Skip.")
        return

    # Load the model and tokenizer
    print(f"Loading model {args.model_name}")
    llm = LLM(args.model_name, tensor_parallel_size=args.num_gpus, dtype="bfloat16", gpu_memory_utilization=0.9, trust_remote_code=True)
    sampling_params = SamplingParams(
        n=args.num_generation, 
        temperature=args.temperature, 
        top_p=args.top_p, 
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    # Load the dataset
    datasets = args.datasets.split(",")
    for dataset_name in datasets:
        dataset = load_dataset(dataset_name, split=args.split)
        # dataset = dataset.filter(lambda example: example['level'] == 'Level 5')
        print(f"Starting from index {args.resume_id} out of {len(dataset)} examples.")
        dataset = dataset.select(range(args.resume_id, len(dataset)))
        if "math" in dataset_name.lower():
            prompt_key = "problem"
            answer_key = "solution"
        elif "aime" in dataset_name.lower() or "amc23" in dataset_name.lower():
            prompt_key = "problem"
            answer_key = "answer"
        dataset = dataset.map(lambda x: prepare_data(x, prompt_key), num_proc=args.dataset_num_proc)

        output_file = dataset_name.split("/")[-1] + '-' + args.split + '-temp_' + str(args.temperature) + "-top_p_" + str(args.top_p) + "-top_k_" + str(args.top_k) + f'{args.comment}.jsonl'
        output_dir = args.output_dir
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        if local_rank == 0 and args.resume_id == 0 and os.path.exists(os.path.join(output_dir, output_file)):
            raise FileExistsError(f"Output file {output_file} already exists.")
        # Create a JSONL file to store the output
        with open(os.path.join(output_dir, output_file), 'w' if args.resume_id == 0 else 'a') as f:
            for i in tqdm(range(0, len(dataset), args.batch_size)):
                batch = dataset[i:i + args.batch_size]
                inputs = batch["prompt"]
                answers = batch[answer_key]

                # Generate the answer
                outputs = llm.generate(inputs, sampling_params=sampling_params, use_tqdm=True)
                results = [[_.outputs[l].text for l in range(len(_.outputs))] for _ in outputs]
                assert len(results[0]) == args.num_generation, f"Number of generations is not equal to {args.num_generation}, got {len(results[0])}"

                # Prepare all outputs for batch tokenization
                flat_outputs = []
                output_mapping = []  # To map back to original indices
                
                for j in range(len(results)):
                    for k in range(args.num_generation):
                        flat_outputs.append(results[j][k])
                        output_mapping.append((j, k))

                # Process the results
                output_idx = 0
                for j, (inp, q, a, r) in enumerate(zip(inputs, batch[prompt_key], answers, results)):
                    for k in range(args.num_generation):
                        qa_pair = {
                            "prompt": inp,
                            "vanilla_response": r[k],
                            "question": q,
                            "answer": a,
                            "question_id": args.resume_id + i + j,
                            "generation_id": k,
                        }
                        qa_pair["response"] = r[k]
                        output_idx += 1
                        if "math" in dataset_name.lower():
                            gold_answer = extract_answer_math(a)
                            pred_answer = extract_answer_math(qa_pair["response"])
                        elif "amc23" in dataset_name.lower() or "aime" in dataset_name.lower():
                            gold_answer = a
                            pred_answer = extract_answer_math(qa_pair["response"])
                        # qa_pair["label"] = pred_answer == gold_answer
                        qa_pair["label"] = math_equal(pred_answer, gold_answer, timeout=True)
                        qa_pair["gold_answer"] = gold_answer
                        qa_pair["pred_answer"] = pred_answer
                        f.write(json.dumps(qa_pair) + '\n')
                f.flush()


if __name__ == "__main__":
    main()
