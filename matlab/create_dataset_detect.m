function data = create_dataset_detect(main_dir, file_ext, percentage)


%category_dirs = dir(main_dir);
    category_dirs_train = dir(strcat(main_dir, '/train_set/split_by_class'));
    category_dirs_test = dir('/media/andrea/Dati2/CV_Proj/handsonbow/dataset/validation/img');
        
    %remove '..' and '.' directories
    category_dirs_train(~cellfun(@isempty, regexp({category_dirs_train.name}, '\.*')))=[];
    category_dirs_train(strcmp({category_dirs_train.name},'split.mat'))=[]; 
    
    category_dirs_test(~cellfun(@isempty, regexp({category_dirs_test.name}, '\.*')))=[];
    category_dirs_test(strcmp({category_dirs_test.name},'split.mat'))=[]; 
    
    if ~exist('percentage','var')
        % third parameter does not exist, so default it to something
        percentage = 100;
    end
    
    for c = 1:(length(category_dirs_train)) 
        imgdirtrain = random_pick_percentage(main_dir,'/train_set/split_by_class',category_dirs_train(c).name, file_ext, percentage);
        imgdirtest = dir(fullfile('/media/andrea/Dati2/CV_Proj/handsonbow/dataset/validation/img', ['*.' file_ext]));
        data(c).n_images = length(imgdirtrain)+length(imgdirtest);
        
        data(c).classname = category_dirs_train(c).name;
        data(c).files = {imgdirtrain(:).name imgdirtest(:).name};
        data(c).train_id = [true(1,length(imgdirtrain)) false(1, data(c).n_images-length(imgdirtrain))];
        data(c).test_id = [false(1,length(imgdirtrain)) true(1, data(c).n_images-length(imgdirtrain))];
    end
end


