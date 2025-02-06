from pca_module import *
from test_module import *
from camera_capture import *    


def main():
    # 1. Build a separate PCA for each folder in "data"
    data_dir = "data"  # Each subfolder in data_dir is a different person
    pca_models, person_list = build_all_pca(data_dir, target_size=(84, 96))
    
    # 2. Test on images in "test" folder
    test_dir = "test"
    test_pca_reconstruction(test_dir, pca_models, person_list, target_size=(84, 96))

if __name__ == "__main__":

    main()


    #Interactive capture
    interactive_camera_capture(
        output_dir="new_user_data",
    )
