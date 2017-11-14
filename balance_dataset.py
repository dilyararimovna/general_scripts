import numpy as np

def balance_multiclass_dataset(dataset_0, labels_0, dataset_1, labels_1, n_classes):
    # Labels one hot
    n_samples_per_classes = np.array([np.count_nonzero(np.array(1. * (labels_0 == i))) for i in range(n_classes)])
    max_n_samples_per_class = np.amax(n_samples_per_classes)
    print("Max number of samples per class: %d" % max_n_samples_per_class)

    result = dataset_0.copy()
    result_labels = labels_0.copy()

    for target_id in range(n_classes):
        target_inds_0 = np.where(labels_0 == target_id)[0]
        target_inds_1 = np.where(labels_1 == target_id)[0]
        target_inds_to_add = target_inds_1[np.random.randint(low=0, high=len(target_inds_1),
        													 size=(max_n_samples_per_class - len(target_inds_0)))]
        result = result.append(dataset_1.loc[target_inds_to_add], ignore_index=True)
        result_labels = result_labels.append(labels_1.loc[target_inds_to_add], ignore_index=True)

    return result, result_labels