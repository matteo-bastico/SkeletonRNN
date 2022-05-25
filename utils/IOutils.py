import numpy as np


def load_data(data_path, scale=1):
    """
    :param data_folder: path to data
    :param scale: skeleton scale factor if any
    :return:
    """
    print("----------- Reading from file %s ------------" % data_path)
    dataset = np.load(data_path, allow_pickle=True)
    skeletons = []
    # Multiple skeletons in dataset, then unpack them
    if dataset.shape[0] > 1:
        for skeleton in dataset:
            skeletons.append(skeleton*scale)
    # Single skeleton
    else:
        skeletons.append(dataset[0, :, :, :]*scale)
    for idx, skeleton in enumerate(skeletons):
        if skeleton.dtype == "object":
            skeletons[idx] = skeleton.astype('float32')
    print("Total number of skeleton sequences in the data folder is: " + str(len(skeletons)))
    return skeletons
