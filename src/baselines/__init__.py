from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from joint_baselines import *
from unsupervised_baselines import *

unsupervised_method_dict = {
    "pca": PCA_learner,
    "seminmf": NMF_learner,
    "ace": ACE_learner,
    "ace_svm": ACE_learner,
    "dictlearn": dictionary_learner,
    "kmeans": kmeans_learner,
}
joint_method_dict = {"ct": concept_transformer_learner}
