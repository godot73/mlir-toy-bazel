import json
import numpy as np


# Loads the outputs from models, parsed from JSON.
# Format:
# [["first prompt", first_output], ["second prompt", second_output], ...]
def load_outputs():
    with open('/tmp/huggingface.json') as file:
        raw_hf = json.load(file)
    with open('/tmp/shark.json') as file:
        raw_shark = json.load(file)
    return (raw_hf, raw_shark)


# Returns only the output values with the propmpts dropped.
def get_values(raw_output):
    return [e[1] for e in raw_output]


def analyze(raw_hf, raw_shark):
    num_prompts = 10
    assert len(raw_hf) == len(raw_shark) == num_prompts
    max_diffs = []
    for prompt_id in range(num_prompts):
        assert raw_hf[prompt_id][0] == raw_shark[prompt_id][0]
        print('Prompt: {}'.format(raw_hf[prompt_id][0]))
        payload_len = len(raw_hf[prompt_id][1][0])
        for i in range(payload_len):
            h = np.array(raw_hf[prompt_id][1][0][i])
            s = np.array(raw_shark[prompt_id][1][0][i])
            max_diff = max(abs(s - h))
            print('\t@{}, max_diff={}'.format(i, max_diff))
            max_diffs.append(max_diff)
    print('\nOverall max_diff={}'.format(max(max_diffs)))


def main():
    raw_hf, raw_shark = load_outputs()
    analyze(raw_hf, raw_shark)


if __name__ == '__main__':
    main()
