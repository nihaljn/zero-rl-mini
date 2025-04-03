"""Evaluate HF models on GSM8K"""
import argparse
import json
import logging
import os
import re

import numpy as np
import randomname
from datasets import Dataset, load_dataset
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


def score_model_outputs(
    sample: dict,
    model_outputs: list[str],
    ks: list[int] = [1]
) -> dict:
    """
    Score outputs for a single sample.

    Args:
        sample (dict): The sample to process
        model_outputs (list[str]): Raw outputs from the model
        ks (list[int]) : Values for which pass@k will be computed

    Returns:
        dict: Results from current evaluation. Includes - {
                    "extracted_responses_raw": [...],
                    "extracted_responses_processed": [...],
                    "gt_response": ...,
                    "response_scores": [...],
                    "pass@k": {k: score for k in [1, 5, ...]}
                }
    """
    extracted_responses_raw, extracted_responses_processed = [], []
    output_dict = {}
    model_input_raw = sample["model_input_raw"]
    
    # postprocess the model output
    for model_output_raw in model_outputs:
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
    """Reduce the results from multiple samples into a single dict."""
    assert len(results) > 0, "No results to reduce"
    reduced_results = {}
    for k in K:
        if f"pass@{k}" in results[0]:
            reduced_results[f"pass@{k}"] = np.mean([
                result[f"pass@{k}"] for result in results
            ]).item()
    return reduced_results


def preprocess_sample(sample: dict, index: int) -> dict:
    """Preprocess sample so that it can be consumed by generation API"""
    output_d = {
        "task_id": f"gsm8k/test/{index}",
        "model_input_raw": prompt_formatter(sample["question"])
    }
    return output_d


def evaluate(
    args: argparse.Namespace, 
    samples: Dataset, 
    generator: Generator,
    generator_return_type: str = "str",
    ks: list[int] = [1]
):
    """"""
    # validation
    if generator_return_type != "dict":
        # TODO: add support later if needed
        raise NotImplementedError
    if args.batch_size < args.n_samples:
        raise ValueError(
            f"batch_size ({args.batch_size}) < n_samples ({args.n_samples})"
        )

    # setup
    num_samples_per_batch = args.batch_size // args.n_samples
    
    # run evaluation
    output_file = os.path.join(args.output_dir, "results.jsonl")
    if os.path.exists(output_file) and not args.overwrite_if_exists:
        mode = "a"
        logger.info(f"Resuming writing results to {output_file}")
    else:
        mode = "w"
        logger.info(f"Writing results to {output_file}")
    with open(output_file, mode) as f:
        pbar = tqdm(
            range(0, len(samples), num_samples_per_batch),
            desc="Generating",
            ncols=100
        )
        for i in pbar:
            # generate
            batch_samples = samples.select(range(i, i + num_samples_per_batch))
            model_inputs_raw = [r["model_input_raw"] for r in batch_samples]
            this_outputs = generator.generate(
                model_inputs_raw, return_type=generator_return_type
            )
            # score generations
            for i in range(num_samples_per_batch):
                sample = batch_samples[i]
                model_outputs_raw = this_outputs["model_outputs_raw"][i]
                output_dict = {
                    "task_id": sample["task_id"],
                    "model_input_raw": sample["model_input_raw"],
                    "model_outputs_raw": model_outputs_raw,
                }
                this_score = score_model_outputs(
                    sample, model_outputs_raw, ks=ks
                )
                output_dict.update(this_score)
                f.write(json.dumps(output_dict) + "\n")


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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--overwrite_if_exists", action="store_true")
    args = parser.parse_args()

    # set up
    seed_everything(args.seed)
    args.exp_name = args.exp_name if args.exp_name is not None else randomname.get_name()
    if args.temperature == 0.0 and args.n_samples != 1:
        logger.warning("n_samples != 1 but temperature = 0.0; setting n_samples = 1")
        args.n_samples = 1
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    logger.info("Running with args: " + str(args))
    if not os.path.exists(args.output_dir):
        logger.info(f"Creating output directory {args.output_dir}")
        os.makedirs(args.output_dir)
    elif args.overwrite_if_exists:
        logger.info(f"Output directory {args.output_dir} already exists; overwriting")
        os.system(f"rm -rf {args.output_dir}/*")
    dataset_config = {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "test",
    }

    # load the model
    generator = Generator("Qwen/Qwen2.5-0.5B", temperature=args.temperature,
                          n_samples=args.n_samples, max_new_tokens=512)

    # write out config            
    logger.info(f"Writing config to {args.output_dir}/config.json")
    config = {k: v for k, v in generator.__dict__.items() 
                   if k not in ["model", "tokenizer"]}
    config.update(dataset_config)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # load the data
    data = load_dataset(**dataset_config)
    if args.max_samples > 0:
        data = data.select(range(args.max_samples))
    # preprocess data
    data = data.map(preprocess_sample, with_indices=True)
    if os.path.exists(os.path.join(args.output_dir, "results.jsonl")):
        # collect completed task ids
        completed_task_ids = set(
            json.loads(l)["task_id"] 
            for l in open(os.path.join(args.output_dir, "results.jsonl"))
        )
        logger.info(f"Found {len(completed_task_ids)} completed task ids")
        # filter out completed task ids
        data = data.filter(
            lambda x: x["task_id"] not in completed_task_ids
        )
    logger.info(f"Using {len(data)} samples for evaluation")
    logger.info("Loaded dataset with config " + str(dataset_config))
    
    # run evaluation
    evaluate(
        args,
        data,
        generator,
        generator_return_type="dict",
        ks=[k for k in K if k <= args.n_samples],
    )

    # reduce scores
    scores = [json.loads(l)["pass@k"] 
              for l in open(os.path.join(args.output_dir, "results.jsonl"))]
    agg_pass_at_k = reduce_pass_at_k(scores)
    logger.info(f"Overall results: {agg_pass_at_k}")
    output_file = os.path.join(args.output_dir, "results.json")
    logger.info(f"Writing overall results to {output_file}")
    with open(output_file, "w") as f:
        json.dump(agg_pass_at_k, f, indent=4)
    logger.info("Done!")


if __name__ == "__main__":
    main()
