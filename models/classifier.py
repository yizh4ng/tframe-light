import numpy as np

from .model import Model
from tframe.utils.maths.confusion_matrix import ConfusionMatrix
from tframe import pedia, DataSet, console
from lambo.gui.vinci.vinci import DaVinci


class Classifier(Model):
  def __init__(self, *args, **kwargs):
    super(Classifier, self).__init__(*args, **kwargs)


  # TODO: Can a model evaluate a dataset? Is it that a dataset can evaludate
  #  a model? No, it should be that someone uses a dataset to evaualte a model
  def evaluate(self, data_set:DataSet, batch_size=1000):
    probs = []
    for batch in data_set.gen_batches(batch_size, is_training=False):
      probs.extend(self.keras_model(batch.features))

    # probs_sorted = np.fliplr(np.sort(probs, axis=-1))
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
    console.write_line(cm.matrix_table(cell_width=4))
    console.show_info('Evaluation Result:')
    console.write_line(cm.make_table(
      decimal=4, class_details=True))

  def show_heatmap(self, gradcam, img, target):
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib import cm
    from tf_keras_vis.utils.scores import CategoricalScore

    # Generate cam with GradCAM++
    cam = gradcam(CategoricalScore(target),
                  img)

    ## Since v0.6.0, calling `normalize()` is NOT necessary.
    # cam = normalize(cam)

    plt.imshow(img)
    heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)  # overlay

  def show_heatmaps_on_dataset(self, data_set:DataSet):
    from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

    da = DaVinci()
    da.objects = data_set

    # Create GradCAM++ object
    da.gradcam = GradcamPlusPlus(self.keras_model,
                              model_modifier=ReplaceToLinear(),
                              clone=True)

    def show_raw(x: DataSet):
      da.imshow(x.features[0], da.axes)

    def show_heatmaps(x: DataSet):
      self.show_heatmap(da.gradcam, x.features[0], np.where(x.targets[0])[0][0])

    da.add_plotter(show_raw)
    da.add_plotter(show_heatmaps)
    da.show()

  def show_activation_maximum(self, dataset):
    labels = np.arange(dataset.num_classes)
    da = DaVinci()
    da.objects = labels
    da.object_titles = [dataset.properties['CLASSES'][label]
                        for label in labels]
    da.activation_maximums = [None for _ in labels]

    from tf_keras_vis.activation_maximization import ActivationMaximization
    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

    activation_maximization = ActivationMaximization(self.keras_model,
                                                     model_modifier=ReplaceToLinear(),
                                                     clone=True)

    def _show_activation_maximum(x, title):
      from tf_keras_vis.utils.scores import CategoricalScore
      from matplotlib import pyplot as plt
      from tf_keras_vis.activation_maximization.callbacks import Progress

      score = CategoricalScore(x)
      if da.activation_maximums[x] is None:
        da.activation_maximums[x] =activation_maximization(score,
                                              callbacks=[Progress()])[0]
      da.imshow(da.activation_maximums[x], title=title)

    da.add_plotter(_show_activation_maximum)
    da.show()

