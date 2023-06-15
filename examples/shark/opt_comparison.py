import collections
import json
import time

from shark.shark_inference import SharkInference
from transformers import AutoTokenizer, OPTForCausalLM

MODEL_NAME = "facebook/opt-1.3b"

PROMPTS = [
    "What is the meaning of life?",
    "Tell me something you don't know.",
    "What does Xilinx do?",
    "What is the mass of earth?",
    "What is a poem?",
    "What is recursion?",
    "Tell me a one line joke.",
    "Who is Gilgamesh?",
    "Tell me something about cryptocurrency.",
    "How did it all begin?",
]

ModelWrapper = collections.namedtuple('ModelWrapper', ['model', 'tokenizer'])


def load_shark_model() -> ModelWrapper:
    vmfb_path = 'opt-1.3b_causallm_30_torch_cpu-sync.vmfb'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    shark_module = SharkInference(mlir_module=None, device='cpu-sync')
    shark_module.load_module(vmfb_path)
    return ModelWrapper(model=shark_module, tokenizer=tokenizer)


def run_shark_model(model_wrapper: ModelWrapper, prompt: str):
    model_inputs = model_wrapper.tokenizer(prompt,
                                           padding="max_length",
                                           max_length=30,
                                           truncation=True,
                                           return_tensors="pt")
    inputs = (
        model_inputs['input_ids'],
        model_inputs['attention_mask'],
    )
    # Generate logits output of OPT model.
    return model_wrapper.model('forward', inputs)


def run_shark():
    model_wrapper = load_shark_model()

    prompt = "What is the meaning of life?"
    logits = run_shark_model(model_wrapper, prompt)

    # Print output logits to validate vs. pytorch + base transformers
    print(logits[0])


def load_huggingface_model() -> ModelWrapper:
    return ModelWrapper(model=OPTForCausalLM.from_pretrained(MODEL_NAME),
                        tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME))


def run_huggingface_model(model_wrapper: ModelWrapper, prompt: str):
    inputs = model_wrapper.tokenizer(prompt, return_tensors="pt")
    return model_wrapper.model.forward(inputs.input_ids,
                                       inputs.attention_mask,
                                       return_dict=False)


def run_huggingface():
    model_wrapper = load_huggingface_model()

    prompt = "What is the meaning of life?"
    logits = run_huggingface_model(model_wrapper, prompt)

    print(logits[0])


def save_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file)


def collect_huggingface_logits():
    t0 = time.time()
    model_wrapper = load_huggingface_model()
    print('--- Took {} seconds to load Huggingface.'.format(
        time.time() - t0))
    results = []
    t0 = time.time()
    for prompt in PROMPTS:
        print('prompt: {}'.format(prompt))
        logits = run_huggingface_model(model_wrapper, prompt)
        results.append([prompt, logits[0].tolist()])
    print('--- Took {} seconds to run Huggingface.'.format(time.time() - t0))
    save_json(results, '/tmp/huggingface.json')


def collect_shark_logits():
    t0 = time.time()
    model_wrapper = load_shark_model()
    print('--- Took {} seconds to load Shark.'.format(time.time() - t0))
    results = []
    t0 = time.time()
    for prompt in PROMPTS:
        print('prompt: {}'.format(prompt))
        logits = run_shark_model(model_wrapper, prompt)
        lst = [e.tolist() for e in logits]
        results.append([prompt, lst])
    print('--- Took {} seconds to run Shark.'.format(time.time() - t0))
    save_json(results, '/tmp/shark.json')


if __name__ == '__main__':
    collect_shark_logits()
    collect_huggingface_logits()
