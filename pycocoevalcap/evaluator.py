from collections import defaultdict
import json
import os
from subprocess import Popen, PIPE
import tempfile
from typing import Any, Dict, List, Union
from mypy_extensions import TypedDict

from .cider import Cider
from .spice import spice


# Some type annotations for better readability
ImageID = int
Caption = str


# Punctuations to be removed from the sentences (PTB style)).
# fmt: off
PUNCTS = [
    "''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", ".", "?", "!",
    ",", ":", "-", "--", "...", ";",
]
# fmt: on


class CocoCaptionsEvaluator(object):
    def __init__(self, gt_annotations: Union[str, List[Any]]):

        # Read annotations from the path (if path is provided).
        if isinstance(gt_annotations, str):
            gt_annotations = json.load(open(gt_annotations))["annotations"]

        # Keep a mapping from image id to a list of captions.
        self.gts: Dict[ImageID, List[Caption]] = defaultdict(list)
        for ann in gt_annotations:
            self.gts[ann["image_id"]].append(ann["caption"])  # type: ignore

        self.gts = tokenize(self.gts)

    def evaluate(self, preds):

        if isinstance(preds, str):
            preds = json.load(open(preds))

        res = {ann["image_id"]: [ann["caption"]] for ann in preds}
        res = tokenize(res)

        # Remove IDs from predictions which are not in GT.
        common_image_ids = self.gts.keys() & res.keys()
        res = {k: v for k, v in res.items() if k in common_image_ids}

        # Add dummy entries for IDs absent in preds, but present in GT.
        for k in self.gts:
            res[k] = res.get(k, [""])

        cider_score, _ = Cider()(self.gts, res)
        spice_score, _ = spice(self.gts, res)

        return {"CIDEr": cider_score, "SPICE": spice_score}


def tokenize(
    image_id_to_captions: Dict[ImageID, List[Caption]]
) -> Dict[ImageID, List[Caption]]:
    r"""
    Given a mapping of image id to a list of corrsponding captions, tokenize
    captions in place according to Penn Treebank Tokenizer. This method assumes
    the presence of Stanford CoreNLP JAR file in directory of this module.
    """
    # Path to the Stanford CoreNLP JAR file.
    CORENLP_JAR = "stanford-corenlp-3.4.1.jar"

    # Prepare data for Tokenizer: write captions to a text file, one per line.
    image_ids = [k for k, v in image_id_to_captions.items() for _ in range(len(v))]
    sentences = "\n".join(
        [c.replace("\n", " ") for k, v in image_id_to_captions.items() for c in v]
    )
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    tmp_file.write(sentences.encode())
    tmp_file.close()

    # fmt: off
    # Tokenize sentences. We use the JAR file for tokenization.
    command = [
        "java", "-cp", CORENLP_JAR, "edu.stanford.nlp.process.PTBTokenizer",
        "-preserveLines", "-lowerCase", tmp_file.name
    ]
    tokenized_captions = (
        Popen(command, cwd=os.path.dirname(os.path.abspath(__file__)), stdout=PIPE)
        .communicate(input=sentences.rstrip())[0]
        .decode()
        .split("\n")
    )
    # fmt: on
    os.remove(tmp_file.name)

    # Map tokenized captions back to their image IDs.
    image_id_to_tokenized_captions: Dict[ImageID, List[Caption]] = defaultdict(list)
    for image_id, caption in zip(image_ids, tokenized_captions):
        image_id_to_tokenized_captions[image_id].append(
            " ".join([w for w in caption.rstrip().split(" ") if w not in PUNCTS])
        )

    return image_id_to_tokenized_captions
