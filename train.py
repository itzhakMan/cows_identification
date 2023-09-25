from classic_cnn import *

def main():
    train_model()
    net = SiameseNetwork()
    net.load_state_dict(torch.load("./trained_siam_net_model_80%_of_data.pt"))
    # net.eval()
    #
    #
    # # ## Some simple testing
    # # The last 3 subjects were held out from the training, and will be used to test. The Distance between each image pair denotes the degree of similarity the model found between the two images. Less means it found more similar, while higher values indicate it found them to be dissimilar.
    #
    # print(face_match(net, "./target"))
    # #visualise_differences(net)


if __name__ == '__main__':
    main()