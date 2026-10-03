"""Reuse probabilities only for factory estimators with identical label rules."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier


def predict_labels_and_probabilities(model,values):
    supported=isinstance(model,(RandomForestClassifier,ExtraTreesClassifier,CalibratedClassifierCV)) or isinstance(model,KNeighborsClassifier) and model.weights=='uniform'
    if supported:
        probabilities=model.predict_proba(values)
        if isinstance(probabilities,np.ndarray) and probabilities.ndim==2 and len(model.classes_)==probabilities.shape[1]:
            return np.asarray(model.classes_)[np.argmax(probabilities,axis=1)],probabilities
        # Multilabel forms are outside the factory contract; preserve their API.
        return model.predict(values),probabilities
    # Logistic/NB/raw-margin GPU rules are retained, including near-tie behavior.
    labels=model.predict(values)
    return labels,model.predict_proba(values) if hasattr(model,'predict_proba') else None
