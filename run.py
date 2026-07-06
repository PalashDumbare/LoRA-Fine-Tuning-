"""Simple launcher for training, evaluation, or inference."""

from qlora_demo.train import train
from qlora_demo.eval import main as eval_main
from qlora_demo.inference import build_model, generate_samples


class QLoRAApp:
    def __init__(self):
        self.actions = {
            "train": self.run_train,
            "eval": self.run_eval,
            "evaluate": self.run_eval,
            "infer": self.run_inference,
            "inference": self.run_inference,
        }

    def run_train(self):
        print("Starting training...")
        train()

    def run_eval(self):
        print("Starting evaluation...")
        eval_main()

    def run_inference(self):
        print("Starting inference...")
        model, tokenizer = build_model()
        generate_samples(model, tokenizer)

    def prompt_action(self):
        while True:
            choice = input("Choose action [train/eval/infer]: ").strip().lower()
            if choice in self.actions:
                return choice
            print("Please enter 'train', 'eval', or 'infer'.")

    def run(self):
        action = self.prompt_action()
        self.actions[action]()


if __name__ == "__main__":
    QLoRAApp().run()
