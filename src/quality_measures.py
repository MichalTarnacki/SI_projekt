import math
from abc import abstractmethod


class QualityMeasures:
    def __init__(self, labels_correct, labels_pred):
        self._labels_correct = labels_correct
        self._labels_pred = labels_pred

        self.precision_in = None
        self.precision_out = None
        self.recall_in = None
        self.recall_out = None
        self.f_in = None
        self.f_out = None

    @abstractmethod
    def count_measures(self):
        pass

    def count_common_measures(self, submeasures):
        if submeasures.tp_in + submeasures.fp_in == 0:
            self.precision_in = 0 if submeasures.tp_in == 0 else math.inf
        else:
            self.precision_in = submeasures.tp_in / (submeasures.tp_in + submeasures.fp_in)

        if submeasures.tp_out + submeasures.fp_out == 0:
            self.precision_out = 0 if submeasures.tp_out == 0 else math.inf
        else:
            self.precision_out = submeasures.tp_out / (submeasures.tp_out + submeasures.fp_out)\

        if submeasures.tp_in + submeasures.fn_in == 0:
            self.recall_in = 0 if submeasures.tp_in == 0 else math.inf
        else:
            self.recall_in = submeasures.tp_in / (submeasures.tp_in + submeasures.fn_in)

        if submeasures.tp_out + submeasures.fn_out == 0:
            self.recall_out = 0 if submeasures.tp_out == 0 else math.inf
        else:
            self.recall_out = submeasures.tp_out / (submeasures.tp_out + submeasures.fn_out)

        if self.precision_in + self.recall_in == 0:
            self.f_in = 0 if 2 * self.precision_in * self.recall_in == 0 else math.inf
        else:
            self.f_in = 2 * self.precision_in * self.recall_in / (self.precision_in + self.recall_in)

        if self.precision_out + self.recall_out == 0:
            self.f_out = 0 if 2 * self.precision_out * self.recall_out == 0 else math.inf
        else:
            self.f_out = 2 * self.precision_out * self.recall_out / (self.precision_out + self.recall_out)

    @abstractmethod
    def count_submeasures(self):
        pass

    def count_common_submeasures(self):
        submeasures = QualitySubMeasures()

        submeasures.tp_in = sum([
            1 if self._labels_pred[i] == self._labels_correct[i] == 'in' else 0
            for i in range(self._labels_pred.shape[0])
        ])
        submeasures.tp_out = sum([
            1 if self._labels_pred[i] == self._labels_correct[i] == 'out' else 0
            for i in range(self._labels_pred.shape[0])
        ])
        submeasures.fp_in = sum([
            1 if self._labels_pred[i] == 'in' and self._labels_correct[i] != 'in' else 0
            for i in range(self._labels_pred.shape[0])
        ])
        submeasures.fp_out = sum([
            1 if self._labels_pred[i] == 'out' and self._labels_correct[i] != 'out' else 0
            for i in range(self._labels_pred.shape[0])
        ])

        return submeasures


class QualityMeasuresTwoClasses(QualityMeasures):
    def __init__(self, labels_correct, labels_pred):
        """
        :param ndarray labels_correct: correct labels ('in' for inhale and 'out' for exhale)
        :param ndarray labels_pred: predicted labels ('in' for inhale and 'out' for exhale)

        Counts accuracy, precisions, recalls and F-measures of the prediction. Counted measures are available
        as fields of the created object.
        """
        super().__init__(labels_correct, labels_pred)

        self.accuracy = None

        self.count_measures()

    def count_measures(self):
        submeasures = self.count_submeasures()

        submeasures.tn_in = submeasures.tp_out
        submeasures.fn_in = submeasures.fp_out
        submeasures.fn_out = submeasures.fp_in

        if submeasures.tp_in + submeasures.fp_in + submeasures.tn_in + submeasures.fn_in == 0:
            self.accuracy = 0 if submeasures.tp_in + submeasures.tn_in == 0 else math.inf
        else:
            self.accuracy = (submeasures.tp_in + submeasures.tn_in) / \
                            (submeasures.tp_in + submeasures.fp_in + submeasures.tn_in + submeasures.fn_in)

        self.count_common_measures(submeasures)

    def count_submeasures(self):
        return self.count_common_submeasures()


class QualityMeasuresThreeClasses(QualityMeasures):
    def __init__(self, labels_correct, labels_pred):
        """
        :param ndarray labels_correct: correct labels ('in' for inhale, 'out' for exhale, any other for none)
        :param ndarray labels_pred: predicted labels ('in' for inhale, 'out' for exhale, any other for none)

        Counts accuracy, precisions, recalls and F-measures of the prediction. Counted measures are available
        as fields of the created object.
        """
        super().__init__(labels_correct, labels_pred)

        self.accuracy_in = None
        self.accuracy_out = None

        self.count_measures()

    def count_measures(self):
        submeasures = self.count_submeasures()

        if submeasures.tp_in + submeasures.fp_in + submeasures.tn_in + submeasures.fn_in == 0:
            self.accuracy_in = 0 if submeasures.tp_in + submeasures.tn_in == 0 else math.inf
        else:
            self.accuracy_in = (submeasures.tp_in + submeasures.tn_in) /\
                               (submeasures.tp_in + submeasures.fp_in + submeasures.tn_in + submeasures.fn_in)

        if submeasures.tp_out + submeasures.fp_out + submeasures.tn_out + submeasures.fn_out == 0:
            self.precision_out = 0 if submeasures.tp_out + submeasures.tn_out == 0 else math.inf
        else:
            self.accuracy_out = (submeasures.tp_out + submeasures.tn_out) / \
                                (submeasures.tp_out + submeasures.fp_out + submeasures.tn_out + submeasures.fn_out)

        self.count_common_measures(submeasures)

    def count_submeasures(self):
        submeasures = self.count_common_submeasures()

        submeasures.tn_in = sum([
            1 if self._labels_pred[i] != 'in' and self._labels_correct[i] != 'in' else 0
            for i in range(self._labels_pred.shape[0])
        ])
        submeasures.tn_out = sum([
            1 if self._labels_pred[i] != 'out' and self._labels_correct[i] != 'out' else 0
            for i in range(self._labels_pred.shape[0])
        ])
        submeasures.fn_in = sum([
            1 if self._labels_pred[i] != 'in' and self._labels_correct[i] == 'in' else 0
            for i in range(self._labels_pred.shape[0])
        ])
        submeasures.fn_out = sum([
            1 if self._labels_pred[i] != 'out' and self._labels_correct[i] == 'out' else 0
            for i in range(self._labels_pred.shape[0])
        ])

        return submeasures


class QualitySubMeasures:
    def __init__(self):
        self.tp_in = None
        self.tn_in = None
        self.fp_in = None
        self.fn_in = None
        self.tp_out = None
        self.tn_out = None
        self.fp_out = None
        self.fn_out = None
