import argparse

from pixelrec.config import SUPPORTED_DATASETS, verify_bundled_data


def main():
    parser = argparse.ArgumentParser(description="Verify bundled PixelRec datasets using SHA-256.")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS)
    args = parser.parse_args()
    for dataset in (args.dataset,) if args.dataset else SUPPORTED_DATASETS:
        verify_bundled_data(dataset)
        print(f"{dataset}: OK")


if __name__ == "__main__":
    main()
