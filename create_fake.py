import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, type=str, required=True, help="")
    parser.add_argument("--output", default=None, type=str, required=True, help="")
    args = parser.parse_args()

    with open(args.input) as fin:
        with open(args.output, "w") as fout:
            for line in fin:
                to_write = " ".join(["::{}".format(word) for word in line.strip().split()]) + "\n"
                fout.write(line)
                fout.write(to_write)


if __name__ == '__main__':
    main()
