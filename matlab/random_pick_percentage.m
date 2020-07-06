function data = random_pick_percentage(main_dir, path, category,file_ext, percentage)
%RANDOM_PICK_PERCENTAGE Summary of this function goes here
%   Detailed explanation goes here

imgdirtrain = dir(fullfile(main_dir,'/train_set/split_by_class',category_dirs_train(c).name, ['*.' file_ext]));

end

