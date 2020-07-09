function data = create_dataset_store(main_dir,isTraining,file_ext)
    category_dirs = dir(main_dir);
    category_dirs_train = dir(strcat(main_dir, '/train_set'));
    category_dirs_test = dir(strcat(main_dir, '/test_set'));
        
    %remove '..' and '.' directories
    category_dirs_train(~cellfun(@isempty, regexp({category_dirs_train.name}, '\.*')))=[];
    category_dirs_train(strcmp({category_dirs_train.name},'split.mat'))=[]; 
    
    category_dirs_test(~cellfun(@isempty, regexp({category_dirs_test.name}, '\.*')))=[];
    category_dirs_test(strcmp({category_dirs_test.name},'split.mat'))=[]; 
    
%     for c = 1:length(category_dirs_train)
%         if isdir(fullfile(main_dir,'/train_set/split_by_class',category_dirs_train(c).name)) && ~strcmp(category_dirs_train(c).name,'.') ...
%                 && ~strcmp(category_dirs_train(c).name,'..')
%             imgdir = dir(fullfile(main_dir,'/train_set/split_by_class',category_dirs_train(c).name, ['*.' file_ext]));
%             %ids = randperm(length(imgdir));
%             data(c).n_images = length(imgdir);
%             data(c).classname = category_dirs_train(c).name;
%             data(c).files = {imgdir(:).name};
%             data(c).train_id = true(1,data(c).n_images);
%             data(c).test_id = false(1,data(c).n_images);
% 
%         end
%     end
%     
%     for d = length(category_dirs_test)
%         if isdir(fullfile(main_dir,'/test_set/split_by_class' ,category_dirs_test(d).name)) && ~strcmp(category_dirs_test(d).name,'.') ...
%                 && ~strcmp(category_dirs_test(d).name,'..')
%             imgdir = dir(fullfile(main_dir,'/test_set/split_by_class',category_dirs_test(d).name, ['*.' file_ext]));
%             %ids = randperm(length(imgdir));
%             data(d).n_images = data(d).n_images + length(imgdir);
%             data(d).classname = category_dirs_test(d).name;
%             data(d).files = {imgdir(:).name};
%             data(d).train_id(c+1:d) = false(1,data(d).n_images);
%             data(d).test_id(c+1:d) = true(1,data(d).n_images);
%         end
%     end
      for c = 1:(length(category_dirs_train))
          imgdirtrain = dir(fullfile(main_dir,'/train_set',category_dirs_train(c).name, ['*.' file_ext]));
          imgdirtest = dir(fullfile(main_dir,'/test_set',category_dirs_test(c).name, ['*.' file_ext]));
          data(c).n_images = (int16(length(imgdirtrain))/100*30) + length(imgdirtest);
          data(c).classname = category_dirs_train(c).name;
          data(c).files = {imgdirtrain(:).name imgdirtest(:).name};
          data(c).train_id = [true(1,length(imgdirtrain)) false(1, data(c).n_images-length(imgdirtrain))];
          data(c).test_id = [false(1,length(imgdirtrain)) true(1, data(c).n_images-length(imgdirtrain))];         
      end
end