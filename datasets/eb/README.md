# Earth-Bench Dataset

## Official download

Download the original Earth-Bench scene data from the official Hugging Face
dataset: [Sssunset/Earth-Bench](https://huggingface.co/datasets/Sssunset/Earth-Bench).

```bash
pip install -U "huggingface-hub[cli]"
huggingface-cli download Sssunset/Earth-Bench \
  --repo-type dataset \
  --local-dir Earth-Bench/benchmark/data
```

Place the downloaded raw files in `Earth-Bench/benchmark/data/`. In particular,
the `question<N>` directories from the dataset must be located directly under
`benchmark/data/`, as shown below:

```text
datasets/eb/
├── README.md
└── Earth-Bench/
    ├── question.json
    ├── extracted_tool_calls_GT.json
    ├── benchmark/
    │   └── data/
    │       ├── question1/
    │       ├── question2/
    │       └── ...
```

For skill extraction, `question.json` and `benchmark/data/` are required.
`extracted_tool_calls_GT.json` is required for the Earth-Bench step-by-step and
end-to-end evaluations.

Configure the Earth-Bench builder with:

```python
train = "SkillEarth/datasets/eb/Earth-Bench/question.json"
train_dir = "SkillEarth/datasets/eb/Earth-Bench"
```
