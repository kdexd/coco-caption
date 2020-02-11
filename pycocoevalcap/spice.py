import os
import subprocess
import json
import numpy as np
import tempfile

# Assumes spice.jar is in the same directory as spice.py.  Change as needed.
SPICE_JAR = "spice-1.0.jar"
TEMP_DIR = "tmp"
CACHE_DIR = "cache"


def spice(gts, res):
    assert sorted(gts.keys()) == sorted(res.keys())
    imgIds = sorted(gts.keys())

    # Prepare temp input file for the SPICE scorer
    input_data = []
    for id in imgIds:
        hypo = res[id]
        ref = gts[id]
        input_data.append({"image_id": id, "test": hypo[0], "refs": ref})

    cwd = os.path.dirname(os.path.abspath(__file__))
    temp_dir = tempfile.mkdtemp()
    json.dump(input_data, open(os.path.join(temp_dir, "input_file.json"), "w"))

    cache_dir = os.path.join(cwd, CACHE_DIR)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # fmt: off
    spice_cmd = [
        "java", "-jar", "-Xmx8G", SPICE_JAR,
        os.path.join(temp_dir, "input_file.json"), "-cache", cache_dir, "-out",
        os.path.join(temp_dir, "output_file.json"), "-subset", "-silent",
    ]
    subprocess.check_call(spice_cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    # Read and process results
    results = json.load(open(os.path.join(temp_dir, "output_file.json")))

    imgId_to_scores = dict()
    spice_scores = []
    for item in results:
        imgId_to_scores[item["image_id"]] = item["scores"]
        spice_scores.append(float_convert(item["scores"]["All"]["f"]))
    average_score = np.mean(np.array(spice_scores))
    scores = []
    for image_id in imgIds:
        # Convert none to NaN before saving scores over subcategories
        score_set = dict()
        for category, score_tuple in imgId_to_scores[image_id].items():
            score_set[category] = {
                k: float_convert(v) for k, v in score_tuple.items()
            }
        scores.append(score_set)
    return average_score, scores


def float_convert(obj):
    try:
        return float(obj)
    except Exception as e:
        return np.nan
