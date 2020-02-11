import copy
from collections import defaultdict
import numpy as np
import math
import os


def precook(s, n=4):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i : i + k])
            counts[ngram] += 1
    return counts


class Cider(object):
    """Main Class to compute the CIDEr metric."""

    def __init__(self, n=4, sigma=6.0):
        # set cider to sum over 1 to 4-grams
        self.n = n
        # set the standard deviation parameter for gaussian penalty
        self.sigma = sigma

    def __call__(self, gts, res):
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus 
        """
        imgIds = sorted(gts.keys())

        crefs = []
        ctest = []
        for id in imgIds:
            hypo = res[id]
            refs = gts[id]

            ctest.append(precook(hypo[0]))
            crefs.append([precook(ref) for ref in refs])

        # compute cider score
        scores = self.compute_cider(crefs, ctest)
        score = np.mean(scores)
        return score, scores

    def compute_cider(self, crefs, ctest):
        def counts2vec(cnts, document_frequency, log_reference_length):
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram, term_freq) in cnts.items():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, document_frequency[ngram]))
                # ngram index
                n = len(ngram) - 1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq) * (log_reference_length - df)
                # compute norm for the vector.  the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            """
            Compute the cosine similarity of two vectors.
            :param vec_hyp: array of dictionary for vector corresponding to hypothesis
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param norm_hyp: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_hyp: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            """
            delta = float(length_hyp - length_ref)
            # measure consine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram, count) in vec_hyp[n].items():
                    # vrama91 : added clipping
                    val[n] += (
                        min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]
                    )

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= norm_hyp[n] * norm_ref[n]

                assert not math.isnan(val[n])
                # vrama91: added a length based gaussian penalty
                val[n] *= np.e ** (-(delta ** 2) / (2 * self.sigma ** 2))
            return val

        # compute idf
        document_frequency = defaultdict(float)
        for refs in crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
                document_frequency[ngram] += 1

        # assert to check document frequency
        assert len(ctest) >= max(document_frequency.values())

        # compute log reference length
        log_reference_length = np.log(float(len(crefs)))

        scores = []
        for test, refs in zip(ctest, crefs):
            # compute vector for test captions
            vec, norm, length = counts2vec(test, document_frequency, log_reference_length)
            # compute vector for ref captions
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(
                    ref, document_frequency, log_reference_length
                )
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score)
            # divide by number of references
            score_avg /= len(refs)
            # multiply score by 10
            score_avg *= 10.0
            # append score of an image to the score list
            scores.append(score_avg)
        return scores
