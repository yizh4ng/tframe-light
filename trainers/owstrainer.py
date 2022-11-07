import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from sklearn.cluster import KMeans
# from k_means_constrained import KMeansConstrained
from tframe.utils.misc import convert_to_one_hot, convert_to_dense_labels
from tframe import console
from sklearn.decomposition import PCA




from.trainer import Trainer



class WeaklySuperviseTrainer(Trainer):
    def __init__(self, *args):
        super(WeaklySuperviseTrainer, self).__init__(*args)
        self.feature_model = None

    def preprocess_features(self, feature, pca=256):
        pca = PCA(n_components=pca)
        pca.fit(feature)
        feature_pca = pca.transform(feature)

        # import faiss
        # mat = faiss.PCAMatrix(feature.shape[1], pca, eigen_power=-0.5)
        # mat.train(feature)
        # assert mat.is_trained
        # feature_pca = mat.apply_py(feature)

        row_sums = np.linalg.norm(feature_pca, axis=1)
        feature_pca = feature_pca / row_sums[:, np.newaxis]
        return feature_pca

    def _int_feature_model(self):
        feature_layer = None
        for layer in self.model.keras_model.layers:
            # if len(layer.output.shape) == 2:
            if isinstance(layer, keras.layers.Dense):
                feature_layer =layer.output
                break
        assert feature_layer is not None

        self.feature_model = tf.keras.Model(inputs=self.model.keras_model.input,
                                            outputs=feature_layer)

    def _update_model_by_batch(self, data_batch):
        target = data_batch.properties['tentative_targets']
        feature = data_batch.features
        loss_dict = {}
        with tf.GradientTape() as tape:
            prediction = self.model.keras_model(feature)
            loss = self.model.loss(prediction, target)
            loss_dict[self.model.loss] = loss
            for metric in self.model.metrics:
                loss_dict[metric] = metric(prediction, target)
        grads = tape.gradient(loss, self.model.keras_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.keras_model.trainable_variables))
        return loss_dict

    def _update_model_by_dataset(self, data_set, rnd):

        # At the beginning of each epoch, set the dataset tentative targets
        # by clustering feautures
        if self.feature_model is None:
            self._int_feature_model()
        if (rnd - 1 ) % 1 == 0:
            features_space = []
            for i, batch in enumerate(data_set.gen_batches(
                self.th.batch_size, updates_per_round=None,
                shuffle=False, is_training=False)):
                features_space.append(self.feature_model(batch.features).numpy())

            features_space = np.concatenate(features_space, axis=0)
            '''Feature pca'''
            # console.show_status('Applying PCA to Feature space...')
            # features_space = self.preprocess_features(features_space, pca=120)

            '''faiss'''
            # console.show_status('Fitting faiss K-means...')
            # import faiss
            # _, d = features_space.shape
            # clus = faiss.Clustering(d, data_set.properties['NUM_CLASSES'])
            # clus.seed = np.random.randint(1234)
            #
            # clus.niter = 20
            # index = faiss.IndexFlatL2(d)
            # console.show_status('faiss is under training...')
            # clus.train(features_space, index)
            # console.show_status('faiss training ends.')
            # _, I = index.search(features_space, 1)
            # labels = np.array([int(n[0]) for n in I])

            '''k-means'''
            console.show_status('Fitting K-means...')
            # kmeans = KMeansConstrained(n_clusters=data_set.properties['NUM_CLASSES'],
            #                            size_min=1000,
            #                            random_state=0).fit(features_space)
            kmeans = KMeans(n_clusters=data_set.properties['NUM_CLASSES'],
                                       # size_min=1000,
                                       random_state=0).fit(features_space)

            labels =  kmeans.labels_

            num_classes = data_set.properties['NUM_CLASSES']
            console.show_status('Swapping targets...')
            total_similarity = []

            for i in range(num_classes):
                max_index_similarity = 0
                swap_target = i
                for j in range(i, num_classes):
                    if 'last_tentative_labels' in list(data_set.properties.keys()):
                        target_index = np.sum(np.argwhere(data_set.properties['last_tentative_labels'] == i), axis=-1)
                    else:
                        target_index = np.sum(np.argwhere(data_set.properties['dense_labels'] == i), axis=-1)
                    tentative_target_index = np.sum(np.argwhere(labels == j), axis=-1)

                    if len(target_index) == 0:
                        index_similarity = 0
                    else:
                        # index_similarity = len(set(target_index.tolist()) & set(tentative_target_index.tolist())) / len(target_index)
                        index_similarity = len(
                            set(target_index.tolist()) & set(
                                tentative_target_index.tolist())) / len(
                            target_index) \
                                           * len(target_index) / len(
                            set(target_index.tolist()).union(set(
                                tentative_target_index.tolist())))

                    if index_similarity > max_index_similarity:
                        max_index_similarity = index_similarity
                        swap_target = j

                total_similarity.append(max_index_similarity)
                swap_target_indices = labels == swap_target
                swapped_target_indices = labels == i
                labels[swap_target_indices] = i
                labels[swapped_target_indices] = swap_target

            console.show_status('Clustering similarity: {}'.format(np.mean(total_similarity)))

            # if 'tentative_labels' in list(data_set.properties.keys()):
            #     data_set.properties['last_tentative_labels'] = data_set.properties['tentative_labels']
            data_set.properties['last_tentative_labels'] = labels
            data_set.properties['tentative_labels'] = labels

            data_set.properties['tentative_targets'] = \
                convert_to_one_hot(labels, num_classes=data_set.properties['NUM_CLASSES'])

            # console.show_status('Sampling balanced dataset...')
            # data_set = data_set.sample_balanced_dataset('tentative_labels')
            # data_set.report()

        assert isinstance(self.model.keras_model, keras.Model)
        flag = True
        for i, layer in enumerate(self.model.keras_model.layers):
            if isinstance(layer, keras.layers.Dense):
                if flag:
                    flag = False
                    continue
                self.model.reset_weights(i)
                 

        for i, batch in enumerate(data_set.gen_batches(
            self.th.batch_size, updates_per_round =self.th.updates_per_round,
            shuffle=self.th.shuffle, is_training=True)):

            self.cursor += 1
            self.counter += 1
            # Update model
            loss_dict = self._update_model_by_batch(batch)
            if np.mod(self.counter - 1, self.th.print_cycle) == 0:
                self._print_progress(i, data_set._dynamic_round_len, rnd, loss_dict)

        self.update_data_set(self.validation_set)
        self.update_data_set(self.test_set)

    def find_swap_groups(self, target, prediction, num_classes):
        console.show_status('Swapping labels...')
        swap_groups = []
        total_similarity = []
        for i in range(num_classes):
            max_index_similarity = 0
            swap_target = i
            for j in range(i, num_classes):
                target_index = np.sum(
                    np.argwhere(target == i), axis=-1)
                tentative_target_index = np.sum(np.argwhere(np.array(prediction) == j), axis=-1)

                if len(target_index) == 0:
                    index_similarity = 0
                else:
                    index_similarity = len(set(target_index.tolist()) & set(
                        tentative_target_index.tolist())) / len(target_index) \
                                       * len(target_index) / len(set(target_index.tolist()).union(set(
                        tentative_target_index.tolist())))

                if index_similarity > max_index_similarity:
                    max_index_similarity = index_similarity
                    swap_target = j

            if i != swap_target:
                swap_groups.append([i, swap_target])
            total_similarity.append(max_index_similarity)
        console.show_status('Clustering similarity: {}'.format(np.mean(total_similarity)))
        return swap_groups


    def update_data_set(self, data_set):
        if self.feature_model is None:
            self._int_feature_model()
        features_space = []
        for i, batch in enumerate(data_set.gen_batches(
            self.th.batch_size, updates_per_round=None,
            shuffle=False, is_training=False)):
            features_space.append(self.model.keras_model(batch.features).numpy())

        features_space = np.concatenate(features_space, axis=0)
        labels = convert_to_dense_labels(features_space)


        num_classes = data_set.properties['NUM_CLASSES']
        swap_groups = self.find_swap_groups(labels, data_set.properties['dense_labels'],
                                            num_classes)

        for swap_group in swap_groups:
            i, swap_target = swap_group
            console.show_status('Swap {} and {}.'.format(i, swap_target))
            swap_target_indices = data_set.properties['dense_labels'] == swap_target
            swapped_target_indices = data_set.properties['dense_labels'] == i
            data_set.properties['dense_labels'][swap_target_indices] = i
            data_set.properties['dense_labels'][swapped_target_indices] = swap_target
            data_set.targets = convert_to_one_hot(data_set.properties['dense_labels'], num_classes)
