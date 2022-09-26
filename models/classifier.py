import numpy as np

from .model import Model
from tframe.utils.maths.confusion_matrix import ConfusionMatrix
from tframe import pedia, DataSet, console
from lambo.gui.vinci.vinci import DaVinci
import tensorflow as tf


class Classifier(Model):
  def __init__(self, *args, **kwargs):
    super(Classifier, self).__init__(*args, **kwargs)


  # TODO: Can a model evaluate a dataset? Is it that a dataset can evaludate
  #  a model? No, it should be that someone uses a dataset to evaualte a model
  def evaluate(self, data_set:DataSet, batch_size=1000):
    console.show_info('Evaluate Confusion Matrix on {}'.format(data_set.name))
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

    # plt.imshow(img)
    heatmap = cam[0]
    # heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
    # plt.imshow(heatmap, cmap='jet', alpha=0.5)  # overlay
    return heatmap

  def show_heatmaps_on_dataset(self, data_set:DataSet):
    from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
    from tf_keras_vis.gradcam import Gradcam
    from tf_keras_vis.saliency import Saliency
    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
    from tf_keras_vis.utils.scores import CategoricalScore

    da = DaVinci()
    da.objects = data_set

    # Create GradCAM++ object
    gradcamplusplus = GradcamPlusPlus(self.keras_model,
                              model_modifier=ReplaceToLinear(),
                              clone=True)
    gradcam = Gradcam(self.keras_model,
                              model_modifier=ReplaceToLinear(),
                              clone=True)
    saliency = Saliency(self.keras_model,
                        model_modifier=ReplaceToLinear(),
                        clone=True)

    def show_raw(x: DataSet):
      da.imshow_pro(x.features[0], title=x.properties['CLASSES'][np.where(x.targets[0])[0][0]])

    def show_heatmap_gradcam(x: DataSet):
      heatmap = self.show_heatmap(gradcam, x.features[0], np.where(x.targets[0])[0][0])
      logits = self.keras_model(x.features)[0]
      confidence = (np.exp(logits)/np.sum(np.exp(logits)))[np.where(x.targets[0])][0]
      da.imshow_pro(heatmap, title='Prediction: '+x.properties['CLASSES'][np.argmax(logits)]
                                   +' Confidence: {:.2f}'.format(confidence))

    def show_heatmap_gradcamplusplus(x: DataSet):
      heatmap = self.show_heatmap(gradcamplusplus, x.features[0], np.where(x.targets[0])[0][0])
      logits = self.keras_model(x.features)[0]
      confidence = (np.exp(logits)/np.sum(np.exp(logits)))[np.where(x.targets[0])][0]
      da.imshow_pro(heatmap, title='Prediction: '+x.properties['CLASSES'][np.argmax(logits)]
                                   +' Confidence: {:.2f}'.format(confidence))

    def show_sliency(x: DataSet):
      logits = self.keras_model(x.features)[0]
      confidence = (np.exp(logits)/np.sum(np.exp(logits)))[np.where(x.targets[0])][0]
      da.imshow_pro(saliency(CategoricalScore(np.where(x.targets[0])[0][0]),
                         x.features[0],
                         smooth_samples=20,
                         smooth_noise=0.20
                         )[0], title='Prediction: '+x.properties['CLASSES'][np.argmax(logits)]
                                   +' Confidence: {:.2f}'.format(confidence))

    da.add_plotter(show_raw)
    da.add_plotter(show_heatmap_gradcam)
    da.add_plotter(show_heatmap_gradcamplusplus)
    da.add_plotter(show_sliency)
    da.show()

  def show_activation_maximum(self, dataset):
    labels = np.arange(dataset.num_classes)
    da = DaVinci()
    da.objects = labels
    da.activation_maximums = [None for _ in labels]

    from tf_keras_vis.activation_maximization import ActivationMaximization
    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

    activation_maximization = ActivationMaximization(self.keras_model,
                                                     model_modifier=ReplaceToLinear(),
                                                     clone=True)
    def _show_activation_maximum(x):
      from tf_keras_vis.utils.scores import CategoricalScore
      from tf_keras_vis.activation_maximization.callbacks import Progress

      score = CategoricalScore(x)
      if da.activation_maximums[x] is None:
        da.activation_maximums[x] =activation_maximization(score,
                                              callbacks=[Progress()])[0]
      da.imshow(da.activation_maximums[x], title=dataset.properties['CLASSES'][x])

    da.add_plotter(_show_activation_maximum)
    da.show()

  def show_feature_space(self, dataset:DataSet, batch_size=1000):
    feature_layer = None
    assert isinstance(self.keras_model, tf.keras.Model)
    for layer in self.keras_model.layers:
      if isinstance(layer, tf.keras.layers.Flatten):
        feature_layer = layer
        break
    assert feature_layer is not None
    new_model = tf.keras.Model(inputs=self.keras_model.input,
                               outputs=feature_layer.output)
    console.show_status('Start reading dataset {}...'.format(dataset.name))
    features = []
    for i, batch in enumerate(dataset.gen_batches(batch_size,
                                                  is_training=False)):
      features.extend(new_model(batch.features))
      console.print_progress(i, dataset.get_round_length(batch_size,
                                                         training=False))
    console.clear_line()
    console.show_status('Finish reading dataset {}.'.format(dataset.name))
    labels = np.argmax(dataset.targets, axis=-1)

    from sklearn.manifold import TSNE
    console.show_status('Start fitting TSNE...')
    tsne = TSNE(n_components=3).fit_transform(features)
    console.show_status('Finish fitting TSNE...')

    def scale_to_01_range(x):

      value_range = (np.max(x) - np.min(x))
      starts_from_zero = x - np.min(x)
      return starts_from_zero / value_range

    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tz = tsne[:, 2]

    # tx = scale_to_01_range(tx)
    # ty = scale_to_01_range(ty)
    # tz = scale_to_01_range(tz)

    classes = dataset.properties['CLASSES']
    colors = [[(np.random.uniform(0.2, 1), np.random.uniform(0.2, 1),
                      np.random.uniform(0.2, 1))] for _ in classes]

    from matplotlib import pyplot as plt
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for idx, c in enumerate(colors):
      indices = [i for i, l in enumerate(labels) if idx == l]
      current_tx = np.take(tx, indices)
      current_ty = np.take(ty, indices)
      current_tz = np.take(tz, indices)
      ax.scatter3D(current_tx, current_ty, current_tz,
                   c=colors[idx] * len(current_tx),
                   label=classes[idx])
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    ax.legend(loc='center left', bbox_to_anchor=(1.07, 0.5))
    plt.show()
