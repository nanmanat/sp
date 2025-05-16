# coding: utf-8
"""Metrics to assess performance on classification task given class prediction

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better
"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Dariusz Brzezinski
# License: MIT

from __future__ import division

import warnings
import logging
import functools

from inspect import getcallargs

import numpy as np
import scipy as sp

from sklearn import metrics
# from sklearn.metrics.classification import (precision_recall_fscore_support)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels

from utils.MAUC import MAUC
from utils.skl_MAUC import skl_MAUC

try:
    from inspect import signature
except ImportError:
    from sklearn.externals.funcsigs import signature

# Implement _prf_divide function since it's not available in the current sklearn version
def _prf_divide(numerator, denominator, metric, modifier, average, warn_for):
    """Performs division and handles divide-by-zero.

    Parameters
    ----------
    numerator : ndarray
        The numerator of the division.
    denominator : ndarray
        The denominator of the division.
    metric : str
        The metric name, used for warning messages.
    modifier : str
        The modifier name, used for warning messages.
    average : str
        The averaging method, used for warning messages.
    warn_for : tuple or set
        A set of metric names to warn for when zero division occurs.

    Returns
    -------
    result : ndarray
        The result of the division, with zeros where the denominator is zero.
    """
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    result = numerator / denominator

    # Set results to 0 where the denominator was 0
    result[mask] = 0.0

    # Warn for zero division if the metric is in warn_for
    if np.any(mask) and metric in warn_for:
        warnings.warn(
            f"{metric} is ill-defined and being set to 0.0 where there is no {modifier} data",
            UserWarning, stacklevel=2)

    return result

# Implement precision_recall_fscore_support function since it's not available in the current sklearn version
def precision_recall_fscore_support(y_true, y_pred, labels=None, pos_label=1,
                                   average=None, warn_for=('precision', 'recall', 'f-score'),
                                   sample_weight=None, zero_division="warn"):
    """Compute precision, recall, F-measure and support for each class

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The F-beta score can be interpreted as a weighted harmonic mean of
    the precision and recall, where an F-beta score reaches its best
    value at 1 and worst score at 0.

    The F-beta score weights recall more than precision by a factor of
    ``beta``. ``beta == 1.0`` means recall and precision are equally important.

    The support is the number of occurrences of each class in ``y_true``.

    If ``pos_label is None`` and in binary classification, this function
    returns the average precision, recall and F-measure if ``average``
    is one of ``'micro'``, ``'macro'``, ``'weighted'`` or ``'samples'``.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    labels : list, optional
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.

    pos_label : str or int, default=1
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : str, default=None
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    warn_for : tuple or set, for internal use
        This determines which warnings will be made in the case that this
        function is being used to return only one of its metrics.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division:
           - recall: when there are no positive labels
           - precision: when there are no positive predictions
           - f-score: both

        If set to "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    precision : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        Precision score.

    recall : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        Recall score.

    fbeta_score : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        F-beta score with beta=1.

    support : None (if average is not None) or array of int, shape =\
        [n_unique_labels]
        The number of occurrences of each label in ``y_true``.

    Notes
    -----
    When ``true positive + false positive == 0``, precision is undefined.
    When ``true positive + false negative == 0``, recall is undefined.
    In such cases, by default the metric will be set to 0, as will f-score,
    and ``UndefinedMetricWarning`` will be raised. This behavior can be
    modified with ``zero_division``.

    References
    ----------
    .. [1] `Wikipedia entry for the Precision and recall
           <https://en.wikipedia.org/wiki/Precision_and_recall>`_

    .. [2] `Wikipedia entry for the F1-score
           <https://en.wikipedia.org/wiki/F1_score>`_

    .. [3] `Discriminative Methods for Multi-labeled Classification
           Advances in Knowledge Discovery and Data Mining (2004), pp. 22-30
           by Shantanu Godbole, Sunita Sarawagi
           <http://www.godbole.net/shantanu/pubs/multilabelsvm-pakdd04.pdf>`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import precision_recall_fscore_support
    >>> y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
    >>> y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
    >>> precision_recall_fscore_support(y_true, y_pred, average='macro')
    (0.22..., 0.33..., 0.26..., None)
    >>> precision_recall_fscore_support(y_true, y_pred, average='micro')
    (0.33..., 0.33..., 0.33..., None)
    >>> precision_recall_fscore_support(y_true, y_pred, average='weighted')
    (0.22..., 0.33..., 0.26..., None)
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    present_labels = unique_labels(y_true, y_pred)

    if average == 'binary':
        if y_type == 'binary':
            if pos_label not in present_labels:
                if len(present_labels) < 2:
                    # Only negative labels
                    return (0., 0., 0., 0)
                else:
                    raise ValueError("pos_label=%r is not a valid label: %r" %
                                     (pos_label, present_labels))
            labels = [pos_label]
        else:
            raise ValueError("Target is %s but average='binary'. Please "
                             "choose another average setting." % y_type)
    elif pos_label not in (None, 1):
        warnings.warn("Note that pos_label (set to %r) is ignored when "
                      "average != 'binary' (got %r). You may use "
                      "labels=[pos_label] to specify a single positive class."
                      % (pos_label, average), UserWarning)

    if labels is None:
        labels = present_labels
        n_labels = None
    else:
        n_labels = len(labels)
        labels = np.hstack([labels, np.setdiff1d(present_labels, labels, assume_unique=True)])

    # Calculate tp_sum, pred_sum, true_sum ###

    if y_type.startswith('multilabel'):
        raise ValueError('Multilabel classification not supported')
    elif average == 'samples':
        raise ValueError("Sample-based precision, recall, fscore is "
                         "not meaningful outside multilabel "
                         "classification. See the accuracy_score instead.")
    else:
        le = LabelEncoder()
        le.fit(labels)
        y_true = le.transform(y_true)
        y_pred = le.transform(y_pred)
        sorted_labels = le.classes_

        # labels are now from 0 to len(labels) - 1 -> use bincount
        tp = y_true == y_pred
        tp_bins = y_true[tp]
        if sample_weight is not None:
            tp_bins_weights = np.asarray(sample_weight)[tp]
        else:
            tp_bins_weights = None

        if len(tp_bins):
            tp_sum = np.bincount(tp_bins, weights=tp_bins_weights, minlength=len(labels))
        else:
            # Pathological case
            true_sum = pred_sum = tp_sum = np.zeros(len(labels))
        if len(y_pred):
            pred_sum = np.bincount(y_pred, weights=sample_weight, minlength=len(labels))
        if len(y_true):
            true_sum = np.bincount(y_true, weights=sample_weight, minlength=len(labels))

        # Retain only selected labels
        indices = np.searchsorted(sorted_labels, labels[:n_labels])
        tp_sum = tp_sum[indices]
        true_sum = true_sum[indices]
        pred_sum = pred_sum[indices]

    if average == 'micro':
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    # Finally, we have all our sufficient statistics. Divide! #
    beta = 1.0

    # Divide, and on zero-division, set scores to 0 and warn:
    precision = _prf_divide(tp_sum, pred_sum, 'precision', 'predicted', average, warn_for)
    recall = _prf_divide(tp_sum, true_sum, 'recall', 'true', average, warn_for)

    # warn for f-score only if zero_division is warn, otherwise it's either 0 or 1
    if beta ** 2 != 0:
        # The score is defined as the harmonic mean of precision and recall
        # calculated as (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        # or equivalently (1 + beta**2) / (1/precision + beta**2/recall)
        # which is equivalent to the weighted arithmetic mean with weights 1/precision and beta**2/recall
        # When precision or recall are 0, the f-score is defined as 0
        f_score = np.where((precision + recall) > 0,
                           (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall),
                           0.)
    else:
        f_score = precision

    # Average the results
    if average == 'weighted':
        weights = true_sum
        if weights.sum() == 0:
            return 0, 0, 0, None
    elif average == 'samples':
        weights = sample_weight
    else:
        weights = None

    if average is not None:
        assert average != 'binary' or len(precision) == 1
        precision = np.average(precision, weights=weights)
        recall = np.average(recall, weights=weights)
        f_score = np.average(f_score, weights=weights)
        true_sum = None  # return no support

    return precision, recall, f_score, true_sum

# Implement _check_targets function since it's not available in the current sklearn version
def _check_targets(y_true, y_pred):
    """Check that y_true and y_pred belong to the same classification task.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.
    y_pred : array-like
        Predicted labels, as returned by a classifier.

    Returns
    -------
    type_true : str
        The type of y_true, one of: 'binary', 'multiclass', 'multilabel-indicator'.
    y_true : array
        The converted y_true.
    y_pred : array
        The converted y_pred.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1:
        raise ValueError("y_true must be 1d array, got %r" % y_true)

    if y_pred.ndim != 1:
        raise ValueError("y_pred must be 1d array, got %r" % y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape, got %r and %r" % 
                         (y_true.shape, y_pred.shape))

    # Check if we have binary or multiclass classification
    classes = np.unique(np.concatenate((y_true, y_pred)))

    if len(classes) <= 2:
        y_type = 'binary'
    else:
        y_type = 'multiclass'

    return y_type, y_true, y_pred


LOGGER = logging.getLogger(__name__)


def sensitivity_specificity_support(y_true,
                                    y_pred,
                                    labels=None,
                                    pos_label=1,
                                    average=None,
                                    warn_for=('sensitivity', 'specificity'),
                                    sample_weight=None):
    """Compute sensitivity, specificity, and support for each class

    The sensitivity is the ratio ``tp / (tp + fn)`` where ``tp`` is the number
    of true positives and ``fn`` the number of false negatives. The sensitivity
    quantifies the ability to avoid false negatives_[1].

    The specificity is the ratio ``tn / (tn + fp)`` where ``tn`` is the number
    of true negatives and ``fn`` the number of false negatives. The specificity
    quantifies the ability to avoid false positives_[1].

    The support is the number of occurrences of each class in ``y_true``.

    If ``pos_label is None`` and in binary classification, this function
    returns the average sensitivity and specificity if ``average``
    is one of ``'weighted'``.

    Read more in the :ref:`User Guide <sensitivity_specificity>`.

    Parameters
    ----------
    y_true : ndarray, shape (n_samples, )
        Ground truth (correct) target values.

    y_pred : ndarray, shape (n_samples, )
        Estimated targets as returned by a classifier.

    labels : list, optional
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.

    pos_label : str or int, optional (default=1)
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : str or None, optional (default=None)
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).
    warn_for : tuple or set, for internal use
        This determines which warnings will be made in the case that this
        function is being used to return only one of its metrics.

    sample_weight : ndarray, shape (n_samples, )
        Sample weights.

    Returns
    -------
    sensitivity : float (if ``average`` = None) or ndarray, \
        shape (n_unique_labels, )

    specificity : float (if ``average`` = None) or ndarray, \
        shape (n_unique_labels, )

    support : int (if ``average`` = None) or ndarray, \
        shape (n_unique_labels, )
        The number of occurrences of each label in ``y_true``.

    References
    ----------
    .. [1] `Wikipedia entry for the Sensitivity and specificity
           <https://en.wikipedia.org/wiki/Sensitivity_and_specificity>`_

    Examples
    --------
    >>> import numpy as np
    >>> from imblearn.metrics import sensitivity_specificity_support
    >>> y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
    >>> y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
    >>> sensitivity_specificity_support(y_true, y_pred, average='macro')
    (0.33333333333333331, 0.66666666666666663, None)
    >>> sensitivity_specificity_support(y_true, y_pred, average='micro')
    (0.33333333333333331, 0.66666666666666663, None)
    >>> sensitivity_specificity_support(y_true, y_pred, average='weighted')
    (0.33333333333333331, 0.66666666666666663, None)

    """
    average_options = (None, 'micro', 'macro', 'weighted', 'samples')
    if average not in average_options and average != 'binary':
        raise ValueError('average has to be one of ' + str(average_options))

    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    present_labels = unique_labels(y_true, y_pred)

    if average == 'binary':
        if y_type == 'binary':
            if pos_label not in present_labels:
                if len(present_labels) < 2:
                    # Only negative labels
                    return (0., 0., 0)
                else:
                    raise ValueError("pos_label=%r is not a valid label: %r" %
                                     (pos_label, present_labels))
            labels = [pos_label]
        else:
            raise ValueError("Target is %s but average='binary'. Please "
                             "choose another average setting." % y_type)
    elif pos_label not in (None, 1):
        warnings.warn("Note that pos_label (set to %r) is ignored when "
                      "average != 'binary' (got %r). You may use "
                      "labels=[pos_label] to specify a single positive class."
                      % (pos_label, average), UserWarning)

    if labels is None:
        labels = present_labels
        n_labels = None
    else:
        n_labels = len(labels)
        labels = np.hstack(
            [labels, np.setdiff1d(
                present_labels, labels, assume_unique=True)])

    # Calculate tp_sum, pred_sum, true_sum ###

    if y_type.startswith('multilabel'):
        raise ValueError('imblearn does not support multilabel')
    elif average == 'samples':
        raise ValueError("Sample-based precision, recall, fscore is "
                         "not meaningful outside multilabel "
                         "classification. See the accuracy_score instead.")
    else:
        le = LabelEncoder()
        le.fit(labels)
        y_true = le.transform(y_true)
        y_pred = le.transform(y_pred)
        sorted_labels = le.classes_

        # labels are now from 0 to len(labels) - 1 -> use bincount
        tp = y_true == y_pred
        tp_bins = y_true[tp]
        if sample_weight is not None:
            tp_bins_weights = np.asarray(sample_weight)[tp]
        else:
            tp_bins_weights = None

        if len(tp_bins):
            tp_sum = np.bincount(
                tp_bins, weights=tp_bins_weights, minlength=len(labels))
        else:
            # Pathological case
            true_sum = pred_sum = tp_sum = np.zeros(len(labels))
        if len(y_pred):
            pred_sum = np.bincount(
                y_pred, weights=sample_weight, minlength=len(labels))
        if len(y_true):
            true_sum = np.bincount(
                y_true, weights=sample_weight, minlength=len(labels))

        # Compute the true negative
        tn_sum = y_true.size - (pred_sum + true_sum - tp_sum)

        # Retain only selected labels
        indices = np.searchsorted(sorted_labels, labels[:n_labels])
        tp_sum = tp_sum[indices]
        true_sum = true_sum[indices]
        pred_sum = pred_sum[indices]
        tn_sum = tn_sum[indices]

    if average == 'micro':
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])
        tn_sum = np.array([tn_sum.sum()])

    # Finally, we have all our sufficient statistics. Divide! #

    with np.errstate(divide='ignore', invalid='ignore'):
        # Divide, and on zero-division, set scores to 0 and warn:

        # Oddly, we may get an "invalid" rather than a "divide" error
        # here.
        specificity = _prf_divide(tn_sum, tn_sum + pred_sum - tp_sum,
                                  'specificity', 'predicted', average,
                                  warn_for)
        sensitivity = _prf_divide(tp_sum, true_sum, 'sensitivity', 'true',
                                  average, warn_for)

    # Average the results

    if average == 'weighted':
        weights = true_sum
        if weights.sum() == 0:
            return 0, 0, None
    elif average == 'samples':
        weights = sample_weight
    else:
        weights = None

    if average is not None:
        assert average != 'binary' or len(specificity) == 1
        specificity = np.average(specificity, weights=weights)
        sensitivity = np.average(sensitivity, weights=weights)
        true_sum = None  # return no support

    return sensitivity, specificity, true_sum


def sensitivity_score(y_true,
                      y_pred,
                      labels=None,
                      pos_label=1,
                      average='binary',
                      sample_weight=None):
    """Compute the sensitivity

    The sensitivity is the ratio ``tp / (tp + fn)`` where ``tp`` is the number
    of true positives and ``fn`` the number of false negatives. The sensitivity
    quantifies the ability to avoid false negatives.

    The best value is 1 and the worst value is 0.

    Read more in the :ref:`User Guide <sensitivity_specificity>`.

    Parameters
    ----------
    y_true : ndarray, shape (n_samples, )
        Ground truth (correct) target values.

    y_pred : ndarray, shape (n_samples, )
        Estimated targets as returned by a classifier.

    labels : list, optional
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average.

    pos_label : str or int, optional (default=1)
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : str or None, optional (default=None)
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    warn_for : tuple or set, for internal use
        This determines which warnings will be made in the case that this
        function is being used to return only one of its metrics.

    sample_weight : ndarray, shape (n_samples, )
        Sample weights.

    Returns
    -------
    specificity : float (if ``average`` = None) or ndarray, \
        shape (n_unique_labels, )

    Examples
    --------
    >>> import numpy as np
    >>> from imblearn.metrics import sensitivity_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> sensitivity_score(y_true, y_pred, average='macro')
    0.33333333333333331
    >>> sensitivity_score(y_true, y_pred, average='micro')
    0.33333333333333331
    >>> sensitivity_score(y_true, y_pred, average='weighted')
    0.33333333333333331
    >>> sensitivity_score(y_true, y_pred, average=None)
    array([ 1.,  0.,  0.])

    """
    s, _, _ = sensitivity_specificity_support(
        y_true,
        y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        warn_for=('sensitivity', ),
        sample_weight=sample_weight)

    return s


def specificity_score(y_true,
                      y_pred,
                      labels=None,
                      pos_label=1,
                      average='binary',
                      sample_weight=None):
    """Compute the specificity

    The specificity is the ratio ``tp / (tp + fn)`` where ``tp`` is the number
    of true positives and ``fn`` the number of false negatives. The specificity
    is intuitively the ability of the classifier to find all the positive
    samples.

    The best value is 1 and the worst value is 0.

    Read more in the :ref:`User Guide <sensitivity_specificity>`.

    Parameters
    ----------
    y_true : ndarray, shape (n_samples, )
        Ground truth (correct) target values.

    y_pred : ndarray, shape (n_samples, )
        Estimated targets as returned by a classifier.

    labels : list, optional
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average.

    pos_label : str or int, optional (default=1)
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : str or None, optional (default=None)
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    warn_for : tuple or set, for internal use
        This determines which warnings will be made in the case that this
        function is being used to return only one of its metrics.

    sample_weight : ndarray, shape (n_samples, )
        Sample weights.

    Returns
    -------
    specificity : float (if ``average`` = None) or ndarray, \
        shape (n_unique_labels, )

    Examples
    --------
    >>> import numpy as np
    >>> from imblearn.metrics import specificity_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> specificity_score(y_true, y_pred, average='macro')
    0.66666666666666663
    >>> specificity_score(y_true, y_pred, average='micro')
    0.66666666666666663
    >>> specificity_score(y_true, y_pred, average='weighted')
    0.66666666666666663
    >>> specificity_score(y_true, y_pred, average=None)
    array([ 0.75,  0.5 ,  0.75])

    """
    _, s, _ = sensitivity_specificity_support(
        y_true,
        y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        warn_for=('specificity', ),
        sample_weight=sample_weight)

    return s


def geometric_mean_score(y_true,
                         y_pred,
                         labels=None,
                         pos_label=1,
                         average='multiclass',
                         sample_weight=None,
                         correction=0.0):
    """Compute the geometric mean

    The geometric mean (G-mean) is the root of the product of class-wise
    sensitivity. This measure tries to maximize the accuracy on each of the
    classes while keeping these accuracies balanced. For binary classification
    G-mean is the squared root of the product of the sensitivity
    and specificity. For multi-class problems it is a higher root of the
    product of sensitivity for each class.

    For compatibility with other imbalance performance measures, G-mean can be
    calculated for each class separately on a one-vs-rest basis when
    ``average != 'multiclass'``.

    The best value is 1 and the worst value is 0. Traditionally if at least one
    class is unrecognized by the classifier, G-mean resolves to zero. To
    alleviate this property, for highly multi-class the sensitivity of
    unrecognized classes can be "corrected" to be a user specified value
    (instead of zero). This option works only if ``average == 'multiclass'``.

    Read more in the :ref:`User Guide <imbalanced_metrics>`.

    Parameters
    ----------
    y_true : ndarray, shape (n_samples, )
        Ground truth (correct) target values.

    y_pred : ndarray, shape (n_samples, )
        Estimated targets as returned by a classifier.

    labels : list, optional
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average.

    pos_label : str or int, optional (default=1)
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : str or None, optional (default=``'multiclass'``)
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    sample_weight : ndarray, shape (n_samples, )
        Sample weights.

    correction: float, optional (default=0.0)
        Substitutes sensitivity of unrecognized classes from zero to a given
        value.

    Returns
    -------
    geometric_mean : float

    Notes
    -----
    See :ref:`sphx_glr_auto_examples_evaluation_plot_metrics.py`.

    References
    ----------
    .. [1] Kubat, M. and Matwin, S. "Addressing the curse of
       imbalanced training sets: one-sided selection" ICML (1997)

    .. [2] Barandela, R., Sánchez, J. S., Garcıa, V., & Rangel, E. "Strategies
       for learning in class imbalance problems", Pattern Recognition,
       36(3), (2003), pp 849-851.

    Examples
    --------
    >>> from imblearn.metrics import geometric_mean_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> geometric_mean_score(y_true, y_pred)
    0.0
    >>> geometric_mean_score(y_true, y_pred, correction=0.001)
    0.010000000000000004
    >>> geometric_mean_score(y_true, y_pred, average='macro')
    0.47140452079103168
    >>> geometric_mean_score(y_true, y_pred, average='micro')
    0.47140452079103168
    >>> geometric_mean_score(y_true, y_pred, average='weighted')
    0.47140452079103168
    >>> geometric_mean_score(y_true, y_pred, average=None)
    array([ 0.8660254,  0.       ,  0.       ])

    """
    if average is None or average != 'multiclass':
        sen, spe, _ = sensitivity_specificity_support(
            y_true,
            y_pred,
            labels=labels,
            pos_label=pos_label,
            average=average,
            warn_for=('specificity', 'specificity'),
            sample_weight=sample_weight)

        LOGGER.debug('The sensitivity and specificity are : %s - %s' %
                     (sen, spe))
        return np.sqrt(sen * spe)
    else:
        present_labels = unique_labels(y_true, y_pred)

        if labels is None:
            labels = present_labels
            n_labels = None
        else:
            n_labels = len(labels)
            labels = np.hstack([labels, np.setdiff1d(present_labels, labels,
                                                     assume_unique=True)])

        le = LabelEncoder()
        le.fit(labels)
        y_true = le.transform(y_true)
        y_pred = le.transform(y_pred)
        sorted_labels = le.classes_

        # labels are now from 0 to len(labels) - 1 -> use bincount
        tp = y_true == y_pred
        tp_bins = y_true[tp]

        if sample_weight is not None:
            tp_bins_weights = np.asarray(sample_weight)[tp]
        else:
            tp_bins_weights = None

        if len(tp_bins):
            tp_sum = np.bincount(tp_bins, weights=tp_bins_weights,
                                 minlength=len(labels))
        else:
            # Pathological case
            true_sum = tp_sum = np.zeros(len(labels))
        if len(y_true):
            true_sum = np.bincount(y_true, weights=sample_weight,
                                   minlength=len(labels))

        # Retain only selected labels
        indices = np.searchsorted(sorted_labels, labels[:n_labels])
        tp_sum = tp_sum[indices]
        true_sum = true_sum[indices]

        recall = _prf_divide(tp_sum, true_sum, "recall", "true", None,
                             "recall")
        recall[recall == 0] = correction

        gmean = sp.stats.gmean(recall)
        # old version of scipy return MaskedConstant instead of 0.0
        if isinstance(gmean, np.ma.core.MaskedConstant):
            return 0.0
        return gmean


def make_index_balanced_accuracy(alpha=0.1, squared=True):
    """Balance any scoring function using the index balanced accuracy

    This factory function wraps scoring function to express it as the
    index balanced accuracy (IBA). You need to use this function to
    decorate any scoring function.

    Only metrics requiring ``y_pred`` can be corrected with the index
    balanced accuracy. ``y_score`` cannot be used since the dominance
    cannot be computed.

    Read more in the :ref:`User Guide <imbalanced_metrics>`.

    Parameters
    ----------
    alpha : float, optional (default=0.1)
        Weighting factor.

    squared : bool, optional (default=True)
        If ``squared`` is True, then the metric computed will be squared
        before to be weighted.

    Returns
    -------
    iba_scoring_func : callable,
        Returns the scoring metric decorated which will automatically compute
        the index balanced accuracy.

    Notes
    -----
    See :ref:`sphx_glr_auto_examples_evaluation_plot_metrics.py`.

    References
    ----------
    .. [1] García, Vicente, Javier Salvador Sánchez, and Ramón Alberto
       Mollineda. "On the effectiveness of preprocessing methods when dealing
       with different levels of class imbalance." Knowledge-Based Systems 25.1
       (2012): 13-21.

    Examples
    --------
    >>> from imblearn.metrics import geometric_mean_score as gmean
    >>> from imblearn.metrics import make_index_balanced_accuracy as iba
    >>> gmean = iba(alpha=0.1, squared=True)(gmean)
    >>> y_true = [1, 0, 0, 1, 0, 1]
    >>> y_pred = [0, 0, 1, 1, 0, 1]
    >>> print(gmean(y_true, y_pred, average=None))
    [ 0.44444444  0.44444444]

    """

    def decorate(scoring_func):
        @functools.wraps(scoring_func)
        def compute_score(*args, **kwargs):
            # Create the list of tags
            tags_scoring_func = getcallargs(scoring_func, *args, **kwargs)
            # check that the scoring function does not need a score
            # and only a prediction
            if ('y_score' in tags_scoring_func or
                'y_prob' in tags_scoring_func or
                    'y2' in tags_scoring_func):
                raise AttributeError('The function {} has an unsupported'
                                     ' attribute. Metric with`y_pred` are the'
                                     ' only supported metrics is the only'
                                     ' supported.')
            # Compute the score from the scoring function
            _score = scoring_func(*args, **kwargs)
            # Square if desired
            if squared:
                _score = np.power(_score, 2)
            # Get the signature of the sens/spec function
            sens_spec_sig = signature(sensitivity_specificity_support)
            # We need to extract from kwargs only the one needed by the
            # specificity and specificity
            params_sens_spec = set(sens_spec_sig._parameters.keys())
            # Make the intersection between the parameters
            sel_params = params_sens_spec.intersection(
                set(tags_scoring_func))
            # Create a sub dictionary
            tags_scoring_func = dict((k, tags_scoring_func[k])
                                     for k in sel_params)
            # Check if the metric is the geometric mean
            if scoring_func.__name__ == 'geometric_mean_score':
                if 'average' in tags_scoring_func:
                    if tags_scoring_func['average'] == 'multiclass':
                        tags_scoring_func['average'] = 'macro'
            # We do not support multilabel so the only average supported
            # is binary
            elif (scoring_func.__name__ == 'accuracy_score' or
                  scoring_func.__name__ == 'jaccard_similarity_score'):
                tags_scoring_func['average'] = 'binary'
            # Create the list of parameters through signature binding
            tags_sens_spec = sens_spec_sig.bind(
                **tags_scoring_func)
            # Call the sens/spec function
            sen, spe, _ = sensitivity_specificity_support(
                *tags_sens_spec.args,
                **tags_sens_spec.kwargs)
            # Compute the dominance
            dom = sen - spe
            return (1. + alpha * dom) * _score

        return compute_score

    return decorate


def classification_report_imbalanced(y_true,
                                     y_pred,
                                     score=None,
                                     num_classes=None,
                                     labels=None,
                                     target_names=None,
                                     sample_weight=None,
                                     digits=3,
                                     alpha=0.1):
    """Build a classification report based on metrics used with imbalanced
    dataset

    Specific metrics have been proposed to evaluate the classification
    performed on imbalanced dataset. This report compiles the
    state-of-the-art metrics: precision/recall/specificity, geometric
    mean, and index balanced accuracy of the
    geometric mean.

    Parameters
    ----------
    y_true : ndarray, shape (n_samples, )
        Ground truth (correct) target values.

    y_pred : ndarray, shape (n_samples, )
        Estimated targets as returned by a classifier.

    labels : list, optional
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average.

    target_names : list of strings, optional
        Optional display names matching the labels (same order).

    sample_weight : ndarray, shape (n_samples, )
        Sample weights.

    digits : int, optional (default=2)
        Number of digits for formatting output floating point values

    alpha : float, optional (default=0.1)
        Weighting factor.

    Returns
    -------
    report : string
        Text summary of the precision, recall, specificity, geometric mean,
        and index balanced accuracy.

    Examples
    --------
    >>> import numpy as np
    >>> from imblearn.metrics import classification_report_imbalanced
    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1] # doctest : +NORMALIZE_WHITESPACE
    >>> target_names = ['class 0', 'class 1', \
    'class 2'] # doctest : +NORMALIZE_WHITESPACE
    >>> print(classification_report_imbalanced(y_true, y_pred, \
    target_names=target_names))
                       pre       rec       spe        f1       geo       iba\
       sup
    <BLANKLINE>
        class 0       0.50      1.00      0.75      0.67      0.87      0.77\
         1
        class 1       0.00      0.00      0.75      0.00      0.00      0.00\
         1
        class 2       1.00      0.67      1.00      0.80      0.82      0.64\
         3
    <BLANKLINE>
    avg / total       0.70      0.60      0.90      0.61      0.66      0.54\
         5
    <BLANKLINE>

    """

    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)

    last_line_heading = 'avg / total'

    if target_names is None:
        target_names = ['%s' % l for l in labels]
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(last_line_heading), digits)

    headers = ["pre", "rec", "spe", "f1", "geo", "iba", "sup"]
    fmt = '%% %ds' % width  # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'

    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

    # Compute the different metrics
    # Precision/recall/f1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        sample_weight=sample_weight)
    # Specificity
    specificity = specificity_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        sample_weight=sample_weight)
    # Geometric mean
    geo_mean = geometric_mean_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        sample_weight=sample_weight)
    # Index balanced accuracy
    iba_gmean = make_index_balanced_accuracy(
        alpha=alpha, squared=True)(geometric_mean_score)
    iba = iba_gmean(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        sample_weight=sample_weight)

    for i, label in enumerate(labels):
        values = [target_names[i]]
        for v in (precision[i], recall[i], specificity[i], f1[i], geo_mean[i],
                  iba[i]):
            values += ["{0:0.{1}f}".format(v, digits)]
        values += ["{0}".format(support[i])]
        report += fmt % tuple(values)

    report += '\n'


    # compute averages
    values = [last_line_heading]
    for v in (np.average(
            precision, weights=support), np.average(
                recall, weights=support), np.average(
                    specificity, weights=support), np.average(
                        f1, weights=support), np.average(
                            geo_mean, weights=support), np.average(
                                iba, weights=support)):
        values += ["{0:0.{1}f}".format(v, digits)]
    values += ['{0}'.format(np.sum(support))]
    report += fmt % tuple(values)

    micropre = np.average(precision, weights=support)
    microrec = np.average(recall, weights=support)
    micromauc = 2.0 * micropre * microrec / float(micropre + microrec)
    report += '\n'
    report += 'micromauc: %.3f\n' % micromauc


    # compute averages
    values = [last_line_heading]
    support = None
    for v in (np.average(
            precision, weights=support), np.average(
                recall, weights=support), np.average(
                    specificity, weights=support), np.average(
                        f1, weights=support), np.average(
                            geo_mean, weights=support), np.average(
                                iba, weights=support)):
        values += ["{0:0.{1}f}".format(v, digits)]
    values += ['{0}'.format(0)]
    report += fmt % tuple(values)

    # compute averages
    values = [last_line_heading]
    support = None
    for v in (np.average(
            precision, weights=support), np.average(
                recall, weights=support), np.average(
                    specificity, weights=support), np.average(
                        f1, weights=support),
                            geometric_mean_score(y_true, y_pred, labels=labels, sample_weight=sample_weight, correction=0.001),
                                np.average(iba, weights=support)):
        values += ["{0:0.{1}f}".format(v, digits)]
    values += ['{0}'.format(0)]
    report += fmt % tuple(values)

    '''
    y_score = [0.99, 0.99, 0.99, 0.99, 0.11]
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=2)
    a = metrics.auc(fpr, tpr)
    report += '\n'
    report += 'auc: %.3f\n' % a
    '''

    '''
    data = list(zip(y_true, score))
    mauc = MAUC(data, num_classes)
    report += '\n'
    report += 'mauc: %.3f\n' % mauc

    y_true_temp = np.zeros((score.shape[0], score.shape[1]), np.int32)
    for ind, y_ in enumerate(y_true):
        y_true_temp[ind][y_] = 1
    y_true = y_true_temp
    micro_auc, macro_auc = skl_MAUC(y_true, score, num_classes)
    report += 'micro_mauc: %.3f\n' % micro_auc
    report += 'macro_mauc: %.3f\n' % macro_auc
    '''

    return report

def classification_report_imbalanced_light(y_true,
                                     y_pred,
                                     score=None,
                                     num_classes=None,
                                     labels=None,
                                     target_names=None,
                                     sample_weight=None,
                                     digits=3,
                                     alpha=0.1):
    """Build a classification report based on metrics used with imbalanced
    dataset

    Specific metrics have been proposed to evaluate the classification
    performed on imbalanced dataset. This report compiles the
    state-of-the-art metrics: precision/recall/specificity, geometric
    mean, and index balanced accuracy of the
    geometric mean.

    Parameters
    ----------
    y_true : ndarray, shape (n_samples, )
        Ground truth (correct) target values.

    y_pred : ndarray, shape (n_samples, )
        Estimated targets as returned by a classifier.

    labels : list, optional
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average.

    target_names : list of strings, optional
        Optional display names matching the labels (same order).

    sample_weight : ndarray, shape (n_samples, )
        Sample weights.

    digits : int, optional (default=2)
        Number of digits for formatting output floating point values

    alpha : float, optional (default=0.1)
        Weighting factor.

    Returns
    -------
    report : string
        Text summary of the precision, recall, specificity, geometric mean,
        and index balanced accuracy.

    Examples
    --------
    >>> import numpy as np
    >>> from imblearn.metrics import classification_report_imbalanced
    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1] # doctest : +NORMALIZE_WHITESPACE
    >>> target_names = ['class 0', 'class 1', \
    'class 2'] # doctest : +NORMALIZE_WHITESPACE
    >>> print(classification_report_imbalanced(y_true, y_pred, \
    target_names=target_names))
                       pre       rec       spe        f1       geo       iba\
       sup
    <BLANKLINE>
        class 0       0.50      1.00      0.75      0.67      0.87      0.77\
         1
        class 1       0.00      0.00      0.75      0.00      0.00      0.00\
         1
        class 2       1.00      0.67      1.00      0.80      0.82      0.64\
         3
    <BLANKLINE>
    avg / total       0.70      0.60      0.90      0.61      0.66      0.54\
         5
    <BLANKLINE>

    """

    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)

    last_line_heading = 'avg / total'

    if target_names is None:
        target_names = ['%s' % l for l in labels]
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(last_line_heading), digits)

    headers = ["pre", "rec", "spe", "f1", "geo", "iba", "sup"]
    fmt = '%% %ds' % width  # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'

    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

    # Compute the different metrics
    # Precision/recall/f1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        sample_weight=sample_weight)
    # Specificity
    specificity = specificity_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        sample_weight=sample_weight)
    # Geometric mean
    geo_mean = geometric_mean_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        sample_weight=sample_weight)
    # Index balanced accuracy
    iba_gmean = make_index_balanced_accuracy(
        alpha=alpha, squared=True)(geometric_mean_score)
    iba = iba_gmean(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        sample_weight=sample_weight)

    for i, label in enumerate(labels):
        values = [target_names[i]]
        for v in (precision[i], recall[i], specificity[i], f1[i], geo_mean[i],
                  iba[i]):
            values += ["{0:0.{1}f}".format(v, digits)]
        values += ["{0}".format(support[i])]
        #report += fmt % tuple(values)

    #report += '\n'


    # compute averages
    values = [last_line_heading]
    for v in (np.average(
            precision, weights=support), np.average(
        recall, weights=support), np.average(
        specificity, weights=support), np.average(
        f1, weights=support), np.average(
        geo_mean, weights=support), np.average(
        iba, weights=support)):
        values += ["{0:0.{1}f}".format(v, digits)]
    values += ['{0}'.format(np.sum(support))]
    report += fmt % tuple(values)

    micropre = np.average(precision, weights=support)
    microrec = np.average(recall, weights=support)
    micromauc = 2.0 * micropre * microrec / float(micropre + microrec)
    report += '\n'
    report += 'micromauc: %.3f\n' % micromauc


    # compute averages
    values = [last_line_heading]
    support = None
    for v in (np.average(
            precision, weights=support), np.average(
        recall, weights=support), np.average(
        specificity, weights=support), np.average(
        f1, weights=support), np.average(
        geo_mean, weights=support), np.average(
        iba, weights=support)):
        values += ["{0:0.{1}f}".format(v, digits)]
    values += ['{0}'.format(0)]
    report += fmt % tuple(values)

    # compute averages
    values = [last_line_heading]
    support = None
    for v in (np.average(
            precision, weights=support), np.average(
        recall, weights=support), np.average(
        specificity, weights=support), np.average(
        f1, weights=support),
              geometric_mean_score(y_true, y_pred, labels=labels, sample_weight=sample_weight, correction=0.001),
              np.average(iba, weights=support)):
        values += ["{0:0.{1}f}".format(v, digits)]
    values += ['{0}'.format(0)]
    report += fmt % tuple(values)

    '''
    y_score = [0.99, 0.99, 0.99, 0.99, 0.11]
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=2)
    a = metrics.auc(fpr, tpr)
    report += '\n'
    report += 'auc: %.3f\n' % a
    '''
    '''
    data = list(zip(y_true, score))
    mauc = MAUC(data, num_classes)
    report += '\n'
    report += 'mauc: %.3f\n' % mauc

    y_true_temp = np.zeros((score.shape[0], score.shape[1]), np.int32)
    for ind, y_ in enumerate(y_true):
        y_true_temp[ind][y_] = 1
    y_true = y_true_temp
    micro_auc, macro_auc = skl_MAUC(y_true, score, num_classes)
    report += 'micro_mauc: %.3f\n' % micro_auc
    report += 'macro_mauc: %.3f\n' % macro_auc
    '''

    return report
