"""Verify real single-token labels without downloading model weights.

Run from inference/: python verify_tokenizer.py
Requires transformers==5.17.0 and network access to public model metadata.
"""
from transformers import AutoTokenizer

from jev_inference.labels import select_labels
from jev_inference.prompt import SYSTEM_PROMPT, format_prompt
from jev_inference.settings import MODEL_ID, MODEL_REVISION


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
    prefix = format_prompt(tokenizer, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "{}"},
    ])
    assert prefix.endswith("<think>\n\n</think>\n\n"), "thinking was not disabled"
    labels = select_labels(tokenizer, prefix)
    assert len(labels) == 255
    print(f"Verified {len(labels)} unique single-token labels: {MODEL_ID}@{MODEL_REVISION}")
    print(f"First labels: {', '.join(label.text for label in labels[:10])}")


if __name__ == "__main__":
    main()
