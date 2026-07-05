import argparse
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

import config
from src.schema_utils import build_schemas


class Predictor:
    def __init__(self, checkpoint_path: str = None, tables_path: str = None, device=None):
        self.device = device or config.DEVICE
        checkpoint_path = checkpoint_path or config.SAVE_PATH
        tables_path     = tables_path or config.TABLES_PATH

        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

        self.db_schemas = build_schemas(tables_path)

        print(f"Model loaded from {checkpoint_path} on {self.device}")

    def predict(self, db_id: str, question: str) -> str:
        schema = self.db_schemas.get(db_id, "")
        src    = f"translate English to SQL: {question} | schema: {schema}"

        inputs = self.tokenizer(
            src,
            return_tensors="pt",
            max_length=config.MAX_SRC_LEN,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                num_beams=config.NUM_BEAMS,
                early_stopping=True,
                no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Generate SQL from an English question.")
    parser.add_argument("--db_id", required=True, help="Database name, e.g. 'people_1'")
    parser.add_argument("--question", required=True, help="English question")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint (.pth)")
    args = parser.parse_args()

    predictor = Predictor(checkpoint_path=args.checkpoint)
    sql = predictor.predict(args.db_id, args.question)

    print(f"\nQ:   {args.question}")
    print(f"SQL: {sql}")


if __name__ == "__main__":
    main()
