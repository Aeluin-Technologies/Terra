"""Dataset processing and loading pipeline for the RSRCC dataset."""

import re
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset

# Regular expressions to parse instructions and responses.
QUESTION_RE = re.compile(
    r"\*\*Question:\*\*(.*?)\*\*Answer:\*\*",
    flags=re.DOTALL,
)

ANSWER_RE = re.compile(
    r"\*\*Answer:\*\*(.*)",
    flags=re.DOTALL,
)


def parse_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Parses raw text in a dataset sample into a structured query format.

    Args:
        sample: A dictionary containing the raw dataset sample fields
            including "text", "before", and "after".

    Returns:
        A dictionary containing parsed components: "before", "after",
        "question", "choices", "answer", and "task".
    """
    text = sample.get("text")

    if not text:
        print(f"Empty text found for sample: {sample.get('before', '')}")
        return {
            "before": sample.get("before", ""),
            "after": sample.get("after", ""),
            "question": "Empty text",
            "choices": None,
            "answer": "",
            "task": "yes_no",
        }

    text = text.replace("\\n", "\n")

    match_question = QUESTION_RE.search(text)
    match_answer = ANSWER_RE.search(text)

    if not match_question or not match_answer:
        print(f"Could not parse text using regular expression: {repr(text)}")

        return {
            "before": sample.get("before", ""),
            "after": sample.get("after", ""),
            "question": "Invalid format in dataset",
            "choices": None,
            "answer": "N/A",
            "task": "yes_no",
        }

    question_content = match_question.group(1).strip()
    answer = match_answer.group(1).strip()

    choices = []
    question_lines = []

    for line in question_content.splitlines():
        line = line.strip()
        if not line:
            continue

        m = re.match(r"\*\*([A-D])\)\*\*\s*(.*)", line)
        if m:
            choices.append(
                {
                    "id": m.group(1),
                    "text": m.group(2).strip(),
                }
            )
        else:
            question_lines.append(line)

    question = "\n".join(question_lines)

    return {
        "before": sample["before"],
        "after": sample["after"],
        "question": question,
        "choices": choices if choices else None,
        "answer": answer,
        "task": "multiple_choice" if choices else "yes_no",
    }


def build_conversation(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Builds a formatted conversation history list for Qwen3-VL processor.

    Args:
        sample: A dictionary containing a dataset sample.

    Returns:
        A list of conversational turn dictionaries matching the OpenAI format.
    """
    parsed = parse_sample(sample)
    prompt = parsed["question"]

    if parsed["choices"] is not None:
        prompt += "\n\n"
        for choice in parsed["choices"]:
            prompt += f"{choice['id']}) {choice['text']}\n"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": parsed["before"],
                },
                {
                    "type": "image",
                    "image": parsed["after"],
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": parsed["answer"],
                }
            ],
        },
    ]

    return messages


class RSRCCIterableDataset(IterableDataset):
    """An iterable wrapper over Hugging Face streaming datasets."""

    def __init__(self, hf_dataset: Any) -> None:
        """Initializes the iterable wrapper dataset.

        Args:
            hf_dataset: The Hugging Face dataset to wrap.
        """
        self.dataset = hf_dataset

    def __iter__(self):
        """Iterates over the dataset, yielding conversation formats.

        Yields:
            A list representing the formatted conversation for a sample.
        """
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            iterator = iter(self.dataset)
        else:
            iterator = self.dataset.shard(
                num_shards=worker.num_workers,
                index=worker.id,
            )

        for sample in iterator:
            try:
                yield build_conversation(sample)
            except Exception as e:
                print(f"Skipping sample due to error: {e}")
                continue


class QwenCollateFn:
    """A callable class that pads tokenized inputs and generates training labels.

    Masks out prompt tokens and only computes loss on the assistant responses.
    """

    def __init__(self, processor: Any) -> None:
        """Initializes the collator with a Qwen3-VL processor.

        Args:
            processor: The processor instance containing the tokenizer.
        """
        self.processor = processor
        self.assistant_prefix = processor.tokenizer.encode(
            "<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        self.im_end_token = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def find_subsequence(self, sequence: List[int], pattern: List[int]) -> int:
        """Finds the starting index of a pattern within a sequence.

        Args:
            sequence: List of input token IDs.
            pattern: List of target token IDs to find.

        Returns:
            The start index of the pattern within the sequence.

        Raises:
            ValueError: If the pattern is not found in the sequence.
        """
        n = len(pattern)
        for i in range(len(sequence) - n + 1):
            if sequence[i : i + n] == pattern:
                return i
        raise ValueError("Assistant prefix not found in target sequence.")

    def build_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Constructs target labels by masking out the prompt tokens with -100.

        Args:
            input_ids: A PyTorch tensor of input token IDs.

        Returns:
            A PyTorch tensor of labels matching input_ids shape.
        """
        labels = torch.full_like(input_ids, -100)
        pad_token_id = self.processor.tokenizer.pad_token_id

        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            try:
                start = self.find_subsequence(ids, self.assistant_prefix)
                answer_start = start + len(self.assistant_prefix)
                answer_end = answer_start

                while (
                    answer_end < len(ids)
                    and ids[answer_end] != self.im_end_token
                    and ids[answer_end] != pad_token_id
                ):
                    answer_end += 1
                labels[i, answer_start:answer_end] = input_ids[
                    i, answer_start:answer_end
                ]
            except ValueError:
                # If prefix not found (e.g. truncated sample), skip masking this item.
                continue

        return labels

    def __call__(self, batch: List[List[Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        """Tokenizes, processes images, pads inputs, and generates label mask.

        Args:
            batch: List of formatted conversations.

        Returns:
            A dictionary containing tensor inputs ready for the model.
        """
        texts = [
            self.processor.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=False,
            )
            for conv in batch
        ]
        images = [
            [
                conv[0]["content"][0]["image"],
                conv[0]["content"][1]["image"],
            ]
            for conv in batch
        ]
        processed = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        processed["labels"] = self.build_labels(processed["input_ids"])
        return processed


def build_dataloader(
    dataset_name: str,
    split: str,
    processor: Any,
    streaming: bool = True,
    batch_size: int = 2,
    seed: int = 42,
    buffer_size: int = 10000,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    """Configures and initializes a PyTorch DataLoader for the RSRCC dataset.

    Args:
        dataset_name: Hugging Face path of the dataset.
        split: The dataset split to load (e.g., 'train', 'test').
        processor: Processor instance for the VLM model.
        streaming: Enable/disable dataset streaming.
        batch_size: Training batch size.
        seed: Shuffling random seed.
        buffer_size: Shuffle buffer size when streaming is active.
        num_workers: Number of subprocess workers.
        pin_memory: Keep tensors in CUDA pinned memory.
        persistent_workers: Maintain loader subprocesses across epochs.
        prefetch_factor: Preload batch count per worker.

    Returns:
        A loaded and configured PyTorch DataLoader.
    """
    ds = load_dataset(dataset_name, split=split, streaming=streaming)

    # TODO: force alternating responses.
    if split == "train":
        if not streaming:
            ds = ds.shuffle(seed=seed)
        else:
            ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

    # NOTE: parse here seems to have no effect with streaming enabled.
    ds = ds.map(parse_sample)

    dataset_wrapper = RSRCCIterableDataset(ds) if streaming else ds
    collate_fn = QwenCollateFn(processor)

    return DataLoader(
        dataset_wrapper,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
