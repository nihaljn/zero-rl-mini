import torch
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from utils import load_model_and_tokenizer


# Ref: https://github.com/hkust-nlp/simpleRL-reason/blob/2d51057f0305cb7b47299affcd513d24e6417a2b/examples/simplelr_math_eval/math_eval.py#L270C28-L270C65
STOP_WORDS = ["<|im_end|>", "<|endoftext|>", "assistant", "user", "_end", "_start"]


# Ref: https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/model_utils.py#L9
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_str: list[str], tokenizer: AutoTokenizer):
        StoppingCriteria.__init__(self)
        self.current_context = []
        self.tokenizer = tokenizer
        self.keywords_str = keywords_str
        self.max_context_to_check = max(
            len(self.tokenizer(word)["input_ids"])
            for word in self.keywords_str
        ) + 2 # some buffer

    def reset(self):
        self.current_context = []
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.Tensor:
        if len(self.current_context) == 0:
            self.current_context = [[] for _ in range(input_ids.shape[0])]

        # Return a boolean tensor indicating which sequences should stop
        should_stop = torch.zeros(input_ids.shape[0], dtype=torch.bool).to(input_ids.device)
        
        for i in range(input_ids.shape[0]):
            _id = input_ids[i][-1].item()
            self.current_context[i].append(_id)

            # Only decode the last few tokens instead of the entire context
            recent_ids = self.current_context[i][-self.max_context_to_check:]
            recent_context = self.tokenizer.decode(recent_ids)
            
            for word in self.keywords_str:
                if word in recent_context:
                    should_stop[i] = True
                    break
        
        # Don't return a single boolean - return a tensor indicating which sequences should stop
        return should_stop


class Generator:
    def __init__(
        self,
        model_name: str,
        temperature: float = 1.0,
        n_samples: int = 1,
        max_new_tokens: int = 512,
        model_dtype: type = torch.bfloat16,
        load_in_half: bool = False,
        use_compile: bool = False
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.do_sample = temperature > 0.0
        self.n_samples = n_samples
        self.max_new_tokens = max_new_tokens
        self.model_dtype = str(model_dtype)
        self.load_in_half = load_in_half
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name, dtype=model_dtype, load_in_half=load_in_half,
            use_compile=use_compile
        )
        self.stop_words = STOP_WORDS
        self.stopping_criteria = StoppingCriteriaList([
            KeywordsStoppingCriteria(self.stop_words, self.tokenizer)
        ])
        self.device = str(self.model.device)
    
    @torch.no_grad()
    def generate(self, prompt: str | list[str], return_type: str = "str") -> list[str] | dict:
        """
        Generate a response from the model.

        Args:
            prompt (str | list[str]): The prompt(s) to generate a response for.
            return_type (str) : Choices - ["str", "dict"].
                                "str" returns a list of list of string responses 
                                (including input)
                                "dict" returns a dict with various metadata

        Returns:
            str: The generated response.
        """
        # tokenize
        tokenized = self.tokenizer(
            prompt, return_tensors="pt", padding=True
        ).to(self.device) 
        
        # generate
        self.stopping_criteria.reset()
        gen_tokens = self.model.generate(
            tokenized.input_ids, # shape: [bs, input_tokens]
            attention_mask=tokenized.attention_mask,
            do_sample=self.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=self.temperature if self.do_sample else 1.0,
            num_return_sequences=self.n_samples,
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=self.stopping_criteria
        ).to("cpu") # shape: [bs*n_samples, input_tokens+new_tokens]
        
        # decode
        gen_text = self.tokenizer.batch_decode(
            gen_tokens, skip_special_tokens=False
        )

        # reshape
        gen_text = [
            gen_text[i*self.n_samples:(i+1)*self.n_samples]
            for i in range(len(prompt))
        ]
        
        # return
        if return_type == "str":
            return gen_text
        elif return_type == "dict":
            output_dict = {
                "model_outputs_raw": gen_text,
            }
            return output_dict
        else:
            raise ValueError(f"Invalid return_type: {return_type}")
