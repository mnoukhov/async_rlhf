import re

import torch
from torch import nn

# from realtreetune
GSM8k_PROMPT = """[MATH_TASK] Problem:
{}

Solution:"""


## Modified from https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    # we can return -1 for errors because answers have to be positive
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


class MathRewardModel(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, postprocessed_responses, label_answer_ids):
        # decode input ids to text, extract answer, set as EOS hidden states
        responses = self.tokenizer.batch_decode(postprocessed_responses)
        pred_answers = [extract_answer(response) for response in responses]

        label_answers = self.tokenizer.batch_decode(label_answer_ids)
        result = [pred == label for pred, label in zip(pred_answers, label_answers)]
        result_tensor = torch.tensor(result, device=label_answer_ids.device)
        # reward of 1 for correct, 0 for incorrect
        reward = result_tensor.to(torch.float)
        return reward

    def modules(self):
        return []

    def to(self, _):
        return self
