from pathlib import Path

from src.train import train


if __name__ == "__main__":
    train(Path(__file__).resolve().parent)
