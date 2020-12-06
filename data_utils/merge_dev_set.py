import os
import json
import fire


def main(meme_dir):
    print("meme_dir: ", meme_dir)
    print("-" * 100)

    meme_val_anno_1 = []
    with open(os.path.join(meme_dir, "dev_seen.jsonl")) as f:
        for line in f:
            meme_val_anno_1.append(json.loads(line))

    meme_val_anno_2 = []
    with open(os.path.join(meme_dir, "dev_unseen.jsonl")) as f:
        for line in f:
            meme_val_anno_2.append(json.loads(line))

    id2val = {}
    for anno in meme_val_anno_1:
        id2val[anno['id']] = anno
    for anno in meme_val_anno_2:
        id2val[anno['id']] = anno

    out_path = os.path.join(meme_dir, "dev_all.jsonl")
    with open(out_path, 'w') as f:
        for anno_line in id2val.values():
            seri_line = json.dumps(anno_line)
            f.write(f"{seri_line}\n")


if __name__ == "__main__":
    fire.Fire(main)