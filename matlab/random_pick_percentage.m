function images = random_pick_percentage(main_dir, path, category,file_ext, percentage) 
    %RANDOM_PICK_PERCENTAGE Summary of this function goes here
    %This function allows us to create a random training set as an arbitrary
    %percentage of the original set 

    imgdirtrain = dir(fullfile(main_dir,'/train_set/split_by_class',category, ['*.' file_ext]));
    [R, C] = size(imgdirtrain);
    ids = randperm(R); %Random permutation
    max_size = int16((length(imgdirtrain)/100)*percentage);
    
    for i = 1:max_size
        images(i) = imgdirtrain(ids(i));
    end

end

