import numpy as np
from .model import Model
from tframe.utils.maths.confusion_matrix import ConfusionMatrix
from tframe import pedia, DataSet, console


class Classifier(Model):
  def __init__(self, *args, **kwargs):
    super(Classifier, self).__init__(*args, **kwargs)


  def evaluate(self, data_set:DataSet):
    probs = self.keras_model(data_set.features)

    probs_sorted = np.fliplr(np.sort(probs, axis=-1))
    if len(probs[0]) == 1: # handle predictor cases
      class_sorted = np.rint(probs)
      class_sorted = np.clip(class_sorted, np.min(data_set.dense_labels),
                             np.max(data_set.dense_labels))
      class_sorted = np.fliplr(class_sorted)
    else:
      class_sorted = np.fliplr(np.argsort(probs, axis=-1))
    preds = class_sorted[:, 0]
    truths = np.ravel(data_set.dense_labels)

    cm = ConfusionMatrix(
      num_classes=data_set.num_classes,
      class_names=data_set.properties.get(pedia.classes, None))
    cm.fill(preds, truths)

    # Print evaluation results
    console.show_info('Confusion Matrix:')
    console.write_line(cm.matrix_table())
    console.show_info('Evaluation Result:')
    console.write_line(cm.make_table(
      decimal=4, class_details=True))
