"""Evaluate HF models on GSM8K"""
import argparse
import json
import logging
import os
import re

import numpy as np
import randomname
from datasets import load_dataset
from tqdm import tqdm

from generator import Generator
from utils import pass_at_k, seed_everything

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger(__name__)
K = [1, 5, 10, 20, 100]


def prompt_formatter(input_text: str) -> str:
    """Hard-coded prompt template defined here"""
    PROMPT_TEMPLATE = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{input}\n"
        "Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    input = PROMPT_TEMPLATE.format(input=input_text)
    return input


def response_extractor(model_output_raw: str) -> int:
    """Extract the response from the model output"""
    # expecting the response to be boxed, so extract the last boxed item in 
    # model output
    pattern = r"\\boxed{(.*)}"
    matches = re.findall(pattern, model_output_raw)
    try:
        return int(matches[-1])
    except:
        return -999999999 # indicating wrong response


def process_sample(
    generator: Generator,
    sample: dict,
    generator_return_type: str = "str",
    ks: list[int] = [1]
) -> dict:
    """
    Process a single sample. Do the following steps:
        1. Preprocess the input
        2. Get the model output
        3. Postprocess the model output
        4. Score the output
        5. Return all the metadata about this run

    Args:
        generator (Generator): The generator to use.
        sample (dict): The sample to process.
        generator_return_type (str) : Return type for the generator.
                                      See `Generator` for more details.
        ks (list[int]) : Values for which pass@k will be computed

    Returns:
        dict: Results from current evaluation. Includes - {
                    "task_id": ...,
                    "input_raw": ...,
                    "model_input_raw: ...,
                    "model_outputs_raw_with_special_tokens": [...],
                    "extracted_responses_raw": [...],
                    "extracted_responses_processed": [...],
                    "gt_response": ...,
                    "response_scores": [...],
                    "pass@k": {k: score for k in [1, 5, ...]}
                }
    """
    # preprocess the input
    question = sample["question"]
    model_input_raw = prompt_formatter(question)
    
    # get the model output
    output_dict = {
        "task_id": sample["task_id"],
        "input_raw": question,
    }
    if generator_return_type != "dict":
        # TODO: add support later if needed
        raise NotImplementedError
    response = generator.generate(model_input_raw, return_type=generator_return_type)
    output_dict.update(response)

    # postprocess the model output
    extracted_responses_raw, extracted_responses_processed = [], []
    for model_output_raw in response["model_outputs_raw_with_special_tokens"]:
        assert model_output_raw.startswith(model_input_raw), \
            f"Model input ({model_input_raw}) and output ({model_output_raw}) " \
             "are not aligned"
        extracted_response = model_output_raw[len(model_input_raw):]
        extracted_response_processed = response_extractor(extracted_response)
        extracted_responses_raw.append(extracted_response)
        extracted_responses_processed.append(extracted_response_processed)
    output_dict.update({
        "extracted_responses_raw": extracted_responses_raw,
        "extracted_responses_processed": extracted_responses_processed,
    })

    # score the reponse
    gt_line = sample["answer"].split("\n")[-1]
    assert gt_line.startswith("#### "), \
        f"Answer is not formatted correctly for {sample['task_id']}"
    gt_response = int(gt_line.split("#### ")[-1])
    response_scores = [
        int(response == gt_response) for response in extracted_responses_processed
    ]
    sample_pass_at_k = pass_at_k(response_scores, ks)
    output_dict.update({
        "gt_response": gt_response,
        "response_scores": response_scores,
        "pass@k": sample_pass_at_k
    })
    
    # return the responses
    return output_dict


def reduce_pass_at_k(results: list[dict]) -> dict:
    """
    Reduce the results from multiple samples into a single dict.
    """
    assert len(results) > 0, "No results to reduce"
    reduced_results = {}
    for k in K:
        if f"pass@{k}" in results[0]["pass@k"]:
            reduced_results[f"pass@{k}"] = np.mean([
                result["pass@k"][f"pass@{k}"] for result in results
            ]).item()
    return reduced_results


def main():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n_samples", type=int, default=1, 
                        help="# sequences while generating")
    args = parser.parse_args()

    # set up
    seed_everything(args.seed)
    args.exp_name = args.exp_name if args.exp_name is not None else randomname.get_name()
    if args.temperature == 0.0 and args.n_samples != 1:
        logger.warning("n_samples != 1 but temperature = 0.0; setting n_samples = 1")
        args.n_samples = 1
    logger.info("Running with args: " + str(args))
    output_dir = os.path.join(args.output_dir, args.exp_name)
    if not os.path.exists(output_dir):
        logger.info(f"Creating output directory {output_dir}")
        os.makedirs(output_dir)

    # load the model
    generator = Generator("Qwen/Qwen2.5-0.5B", temperature=args.temperature,
                          n_samples=args.n_samples, max_new_tokens=512)

    # load the data
    dataset_config = {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "test",
    }
    data = load_dataset(**dataset_config)
    if args.max_samples > 0:
        data = data.select(range(args.max_samples))
    logger.info(f"Using {len(data)} samples for evaluation")
    # add task identifiers
    data = data.map(lambda x, i: {"task_id": i}, with_indices=True)
    logger.info("Loaded dataset with config " + str(dataset_config))
    
    # map each input to output
    output_file = os.path.join(output_dir, "results.jsonl")
    logger.info(f"Writing results to {output_file}")
    outputs = []
    with open(output_file, "w") as f:
        for sample in tqdm(data, ncols=100, desc="Generating"):
            output = process_sample(
                generator,
                sample,
                generator_return_type="dict",
                ks=[k for k in K if k <= args.n_samples]
            )
            outputs.append(output)
            logger.debug(f"Processed sample {sample['task_id']}")
            f.write(json.dumps(output) + "\n")
    
    # write out config            
    logger.info(f"Writing config to {output_dir}/config.json")
    config = generator.__dict__
    del config["model"]
    del config["tokenizer"]
    config.update(dataset_config)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # write out overall results
    agg_pass_at_k = reduce_pass_at_k(outputs)
    logger.info(f"Overall results: {agg_pass_at_k}")
    logger.info(f"Writing overall results to {output_dir}/results.json")
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(agg_pass_at_k, f, indent=4)
    logger.info("Done!")


if __name__ == "__main__":
    main()
