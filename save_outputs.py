import numpy as np
from sklearn.decomposition import PCA
from classic_cnn import *
from dim_reduction import *
from numpy import savetxt
from sklearn.preprocessing import StandardScaler
from siam_with_resnet import *
import torch.nn.functional as F

def save_net_outputs(weights_path,
                     input_folder_path,
                     output_name_csv,
                     labels_name_csv,
                     in_net):
    net = in_net
    # net = SiameseResLstmNetwork()
    net.load_state_dict(torch.load(weights_path))
    net.eval()

    output_arr, label_arr = get_net_outputs(net,input_folder_path)
    # output_arr_len = len(output_arr)
    # color_arr = np.ones(shape=(1,output_arr_len))


    savetxt(output_name_csv, output_arr)
    savetxt(labels_name_csv,label_arr )


    # sklearn_pca2d = PCA(n_components=2)
    # plot_2d(sklearn_pca2d.fit_transform(output_arr), label_arr, "sklearn PCA 2D on test data")
    #
    # sklearn_pca3d = PCA(n_components=3)
    # plot_3d(sklearn_pca3d.fit_transform(output_arr), label_arr, "sklearn PCA 3D on test data")
    #
    # scaler = StandardScaler()
    # output_arr_scaled = scaler.fit_transform(output_arr)
    # plot_2d(sklearn_pca2d.fit_transform(output_arr_scaled), label_arr, "sklearn PCA 2D on test data scaled")
    # plot_3d(sklearn_pca3d.fit_transform(output_arr_scaled), label_arr, "sklearn PCA 3D on test data scaled")

    # kernel = lambda x, y:  F.pairwise_distance(x, y, keepdim=True)
    # KPCA_ = KPCA(sklearn_pca3d.fit_transform(output_arr), 3, kernel)
    # plot_3d(KPCA_, color_arr, "KPCA")
    # # visualise_differences(net)

def main():
    save_net_outputs("./trained_siam_net_model_cows.pt",
                     "./data/cows/cow_dataset/cows/Sub-levels/Identification/Test",
                     'results/NET_OUTPUT_testing_cows_new.csv',
                     'results/labels_testing_cows_new.csv')

if __name__ == '__main__':
    main()