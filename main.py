import argparse
import pytorch_lightning as pl
from src.experiment import Experiment

def main(args):
    pl.seed_everything(46)

    trainer = pl.Trainer(
        max_epochs = args.epochs,
    )

    exp = Experiment(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    exp.fit(trainer)
    exp.save(trainer, args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--save_path", type=str)

    args = parser.parse_args()

    main(args)
