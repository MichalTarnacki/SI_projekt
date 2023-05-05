class QualityMeasures:
    def __init__(self, labels_correct, labels_pred):
        """
        Counts accuracy, precisions, recalls and F-measures of the prediction.

        Arguments:
        labels_correct: ndarray of correct labels ('in' for inhale and 'out' for exhale)
        labels_pred: ndarray of predicted labels ('in' for inhale and 'out' for exhale)

        Counted measures are available as fields of the created object.
        """
        self.accuracy = None
        self.precision_in = None
        self.precision_out = None
        self.recall_in = None
        self.recall_out = None
        self.f_in = None
        self.f_out = None

        self.count_measures(labels_correct, labels_pred)

    def count_measures(self, labels_correct, labels_pred):
        true_pos_in, true_pos_out, false_pos_in, false_pos_out = self.get_poses_and_negs(labels_correct, labels_pred)

        true_neg_in = true_pos_out
        false_neg_in = false_pos_out
        false_neg_out = false_pos_in

        self.accuracy =(true_pos_in + true_neg_in) / (true_pos_in + false_pos_in + true_neg_in + false_neg_in)
        self.precision_in = true_pos_in / (true_pos_in + false_pos_in)
        self.precision_out = true_pos_out / (true_pos_out + false_pos_out)
        self.recall_in = true_pos_in / (true_pos_in + false_neg_in)
        self.recall_out = true_pos_out / (true_pos_out + false_neg_out)
        self.f_in = 2 * self.precision_in * self.recall_in / (self.precision_in + self.recall_in)
        self.f_out = 2 * self.precision_out * self.recall_out / (self.precision_out + self.recall_out)

    @staticmethod
    def get_poses_and_negs(labels_correct, labels_pred):
        true_pos_in = sum([
            1 if labels_pred[i] == labels_correct[i] == 'in' else 0
            for i in range(labels_pred.shape[0])
        ])
        true_pos_out = sum([
            1 if labels_pred[i] == labels_correct[i] == 'out' else 0
            for i in range(labels_pred.shape[0])
        ])

        false_pos_in = sum([
            1 if labels_pred[i] == 'in' and labels_correct[i] == 'out' else 0
            for i in range(labels_pred.shape[0])
        ])
        false_pos_out = sum([
            1 if labels_pred[i] == 'out' and labels_correct[i] == 'in' else 0
            for i in range(labels_pred.shape[0])
        ])

        return true_pos_in, true_pos_out, false_pos_in, false_pos_out
