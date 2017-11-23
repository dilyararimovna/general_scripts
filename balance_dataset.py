import numpy as np

def balance_multiclass_dataset(dataset_0, labels_0, dataset_1, labels_1, n_classes):
    # Labels one hot
    n_samples_per_classes = np.array([np.count_nonzero(np.array(1. * (labels_0 == i))) 
    	for i in range(n_classes)])
    print('Initial number of samples per class:', n_samples_per_classes)
    max_n_samples_per_class = np.amax(n_samples_per_classes)
    print("Max number of samples per class: %d" % max_n_samples_per_class)

    result = list(dataset_0.copy())
    result_labels = list(labels_0.copy())
    for target_id in range(n_classes):

        target_inds_0 = np.where(labels_0 == target_id)[0]
        target_inds_1 = np.where(labels_1 == target_id)[0]
        if (len(target_inds_1) == 0):
        	print("Warning: Nothing to add from the second dataset of label %d" % target_id)
        	continue
        target_inds_to_add = target_inds_1[np.random.randint(low=0, high=len(target_inds_1),
        													 size=(max_n_samples_per_class 
        													 	- len(target_inds_0)))]
        result.extend(dataset_1[target_inds_to_add])
        result_labels.extend(labels_1[target_inds_to_add])

    return result, result_labels