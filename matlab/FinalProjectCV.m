%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Final Project CV 2020                                                  %
%  Find the position of a person into the store given a classifier ...    %
%                                                                         %
%  Students:                                                              %
%  Daniele Ortu   <danieleortu8@gmail.com>                                %
%  Andrea Atzori  <atzoriandrea@outlook.it>                               %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clean command windows, workspace 
clear;
clc;

% Close secondary windows 
close all;

% Inizializations
cols = {'rgbFile','depthFile','x', 'y', 'u', 'v', 'class' };
path = '/media/andrea/Dati2/CV_Proj/handsonbow/dataset/train_set/train_set.txt';
base_path = '/media/andrea/Dati2/CV_Proj/handsonbow/dataset/train_set/';
data = readtable(path,'Delimiter', ' ', 'HeaderLines', 0, 'ReadVariableNames',true );
data = data(:,1:7);

libsvmpath = [fullfile('lib','libsvm-3.11','matlab')];
addpath(libsvmpath);

% Do or not some operations
importTrainingSetData = 1;
importTestSetData = 0;
do_createfolders = 0;
do_copyimages = 0;
do_split_sets = 1;
do_feat_extraction = 1;

% type operation
desc_name = 'dsift';

% Image file extension
file_ext='jpg';

% nameColumns = {'rgbFile', 'depthFile', 'x', 'y', 'u', 'v', 'class'};

%Train set
dataTrain = readtable(fullfile('dataset', 'train_set', 'train_set.txt'));
dataTrain = dataTrain(:,1:7);
dataTrain = table2cell(dataTrain);

%Test set
dataTest = readtable(fullfile('dataset', 'test_set', 'test_set.txt'));
dataTest = dataTest(:,1:7);
dataTest = table2cell(dataTest);

if do_createfolders
    for i = 1:16
        mkdir('/media/andrea/Dati2/CV_Proj/handsonbow/dataset/train_set/split_by_class', int2str(i));
        mkdir('/media/andrea/Dati2/CV_Proj/handsonbow/dataset/test_set/split_by_class', int2str(i));
    end
end

if do_copyimages
    %Training set
    [M, N] = size(dataTrain);
    for i = 1 : M
        class = testMatrix(i,7);
        file_RGB = strcat('/media/andrea/Dati2/CV_Proj/handsonbow/dataset/train_set/train_RGB/', testMatrix(i,1));
        dest = strcat('/media/andrea/Dati2/CV_Proj/handsonbow/dataset/train_set/split_by_class/', num2str(cell2mat(class)));
        copyfile(string(file_RGB), dest)
    end
    
    %Test set
    [M, N] = size(dataTest);
    for i = 1 : M
        class = testMatrix(i,7);
        file_RGB = strcat('/media/andrea/Dati2/CV_Proj/handsonbow/dataset/test_set/test_RGB/', testMatrix(i,1));
        dest = strcat('/media/andrea/Dati2/CV_Proj/handsonbow/dataset/test_set/split_by_class/', num2str(cell2mat(class)));
        copyfile(string(file_RGB), dest)
    end
end

% Create a new dataset split for train
file_split = 'split.mat';
if do_split_sets    
    %data = create_dataset_split_structure(fullfile(basepath, 'dataset/train_set/split_by_class', dataset_dir),50,0 ,file_ext);
    data = create_dataset_store(fullfile(basepath, 'dataset'), 1, file_ext)
    save(fullfile(basepath,'dataset',dataset_dir,file_split),'data');
else
    load(fullfile(basepath,'dataset',dataset_dir,file_split));
end

classes = {data.classname}; % create cell array of class name strings

% Extract SIFT features fon training and test images
if do_feat_extraction  
    extract_sift_features(fullfile(basepath, 'dataset/test_set/split_by_class'),desc_name)    
    extract_sift_features(fullfile(basepath, 'dataset/train_set/split_by_class'),desc_name)    
end


% Convert images of the training set into BoVW representation


% Load pre-computed SIFT features for training images

% The resulting structure array 'desc' will contain one
% entry per images with the following fields:
%  desc(i).r :    Nx1 array with y-coordinates for N SIFT features
%  desc(i).c :    Nx1 array with x-coordinates for N SIFT features
%  desc(i).rad :  Nx1 array with radius for N SIFT features
%  desc(i).sift : Nx128 array with N SIFT descriptors
%  desc(i).imgfname : file name of original image

lasti=1;
for i = 1:length(data)
     images_descs = get_descriptors_files(data,i,file_ext,desc_name,'train');
     for j = 1:length(images_descs) 
        fname = fullfile(basepath,'dataset/train_set/split_by_class',dataset_dir,data(i).classname,images_descs{j});
        fprintf('Loading %s \n',fname);
        tmp = load(fname,'-mat');
        tmp.desc.class=i;
        tmp.desc.imgfname=regexprep(fname,['.' desc_name],'.jpg');
        desc_train(lasti)=tmp.desc;
        desc_train(lasti).sift = single(desc_train(lasti).sift);
        lasti=lasti+1;
     end
end


% Visualize SIFT features for training images
if (visualize_feat && have_screen)
    nti=10;
    fprintf('\nVisualize features for %d training images\n', nti);
    imgind=randperm(length(desc_train));
    for i=1:nti
        d=desc_train(imgind(i));
        clf, showimage(imread(strrep(d.imgfname,'_train','')));
        x=d.c;
        y=d.r;
        rad=d.rad;
        showcirclefeaturesrad([x,y,rad]);
        title(sprintf('%d features in %s',length(d.c),d.imgfname));
        pause
    end
end


% Load pre-computed SIFT features for test images 

lasti=1;
for i = 1:length(data)
     images_descs = get_descriptors_files(data,i,file_ext,desc_name,'test');
     for j = 1:length(images_descs) 
        fname = fullfile(basepath,'dataset/test_set/split_by_class',dataset_dir,data(i).classname,images_descs{j});
        fprintf('Loading %s \n',fname);
        tmp = load(fname,'-mat');
        tmp.desc.class=i;
        tmp.desc.imgfname=regexprep(fname,['.' desc_name],'.jpg');
        desc_test(lasti)=tmp.desc;
        desc_test(lasti).sift = single(desc_test(lasti).sift);
        lasti=lasti+1;
     end
end


% Build visual vocabulary using k-means 

if do_form_codebook
    fprintf('\nBuild visual vocabulary:\n');

    % concatenate all descriptors from all images into a n x d matrix 
    DESC = [];
    labels_train = cat(1,desc_train.class);
    for i=1:length(data)
        desc_class = desc_train(labels_train==i);
        randimages = randperm(length(desc_class));
        randimages=randimages(1: int16(data(i).n_images/100*15));
        DESC = vertcat(DESC,desc_class(randimages).sift);
    end

    % sample random M (e.g. M=20,000) descriptors from all training descriptors
    r = randperm(size(DESC,1));
    r = r(1:min(length(r),nfeat_codebook));

    DESC = DESC(r,:);

    % run k-means
    K = nwords_codebook; % size of visual vocabulary
    fprintf('running k-means clustering of %d points into %d clusters...\n',...
        size(DESC,1),K)
    % input matrix needs to be transposed as the k-means function expects 
    % one point per column rather than per row

    % form options structure for clustering
    cluster_options.maxiters = max_km_iters;
    cluster_options.verbose  = 1;

    [VC] = kmeans_bo(double(DESC),K,max_km_iters);%visual codebook
    VC = VC';%transpose for compatibility with following functions
    clear DESC;
end


% K-means descriptor quantization means assignment of each feature
% descriptor with the identity of its nearest cluster mean, i.e.
% visual word. Your task is to quantize SIFT descriptors in all
% training and test images using the visual dictionary 'VC'
% constructed above.


if do_feat_quantization
    fprintf('\nFeature quantization (hard-assignment)...\n');
    for i=1:length(desc_train)  
      
      dmat=eucliddist(desc_train(i).sift,VC);
      [mv, visword] = min(dmat, [], 2);
      % save feature labels
      desc_train(i).visword = visword;
      desc_train(i).quantdist = mv;
    end

    for i=1:length(desc_test)    
      
      dmat=eucliddist(desc_test(i).sift,VC);
      [mv, visword] = min(dmat, [], 2);
      
      % save feature labels
      desc_test(i).visword = visword;
      desc_test(i).quantdist = mv;
    end
end



% Visualize visual words (i.e. clusters)
%  To visually verify feature quantization computed above, we can show 
%  image patches corresponding to the same visual word. 

if (visualize_words && have_screen)
    figure;
    %num_words = size(VC,1) % loop over all visual word types
    num_words = 10;
    fprintf('\nVisualize visual words (%d examples)\n', num_words);
    for i=1:num_words
      patches={};
      for j=1:length(desc_train) % loop over all images
        d=desc_train(j);
        ind=find(d.visword==i);
        if length(ind)
          %img=imread(strrep(d.imgfname,'_train',''));
          img=rgb2gray(imread(d.imgfname));
          
          x=d.c(ind); y=d.r(ind); r=d.rad(ind);
          bbox=[x-2*r y-2*r x+2*r y+2*r];
          for k=1:length(ind) % collect patches of a visual word i in image j      
            patches{end+1}=cropbbox(img,bbox(k,:));
          end
        end
      end
      % display all patches of the visual word i
      clf, showimage(combimage(patches,[],1.5))
      title(sprintf('%d examples of Visual Word #%d',length(patches),i))
      pause
    end
end






% Represent each image by the normalized histogram of visual
% word labels of its features. Compute word histogram H over 
% the whole image, normalize histograms w.r.t. L1-norm.


N = size(VC,1); % number of visual words

for i=1:length(desc_train) 
    visword = desc_train(i).visword;
    H = histc(visword,[1:nwords_codebook]);
  
    % normalize bow-hist (L1 norm)
    if norm_bof_hist
        H = H/sum(H);
    end
  
    % save histograms
    desc_train(i).bof=H(:)';
end

for i=1:length(desc_test) 
    visword = desc_test(i).visword;
    H = histc(visword,[1:nwords_codebook]);
  
    % normalize bow-hist (L1 norm)
    if norm_bof_hist
        H = H/sum(H);
    end
  
    % save histograms
    desc_test(i).bof=H(:)';
end



%LLC Coding
if do_svm_llc_linar_classification
    for i=1:length(desc_train)
        disp(desc_train(i).imgfname);
        desc_train(i).llc = max(LLC_coding_appr(VC,desc_train(i).sift)); %max-pooling
        desc_train(i).llc=desc_train(i).llc/norm(desc_train(i).llc); %L2 normalization
    end
    for i=1:length(desc_test) 
        disp(desc_test(i).imgfname);
        desc_test(i).llc = max(LLC_coding_appr(VC,desc_test(i).sift));
        desc_test(i).llc=desc_test(i).llc/norm(desc_test(i).llc);
    end
end
%end LLC coding

save(fullfile(basepath,'dataset','desc_test.mat'),'desc_test','-v7.3' );
save(fullfile(basepath,'dataset','desc_train.mat'),'desc_train','-v7.3' );



% Concatenate bof-histograms into training and test matrices 
bof_train=cat(1,desc_train.bof);
bof_test=cat(1,desc_test.bof);
if do_svm_llc_linar_classification
    llc_train = cat(1,desc_train.llc);
    llc_test = cat(1,desc_test.llc);
end

% Construct label Concatenate bof-histograms into training and test matrices 
labels_train=cat(1,desc_train.class);
labels_test=cat(1,desc_test.class);

% Induction of the classifier

% NN classification 

if do_L2_NN_classification
    % Compute L2 distance between BOFs of test and training images
    bof_l2dist=eucliddist(bof_test,bof_train);
    
    % Nearest neighbor classification (1-NN) using L2 distance
    [mv,mi] = min(bof_l2dist,[],2);
    bof_l2lab = labels_train(mi);
    
    method_name='NN L2';
    acc=sum(bof_l2lab==labels_test)/length(labels_test);
    fprintf('\n*** %s ***\nAccuracy = %1.4f%% (classification)\n',method_name,acc*100);
   
    % Compute classification accuracy
    compute_accuracy(data,labels_test,bof_l2lab,classes,method_name,desc_test,...
                      visualize_confmat & have_screen,... 
                      visualize_res & have_screen);
end

% Knn to find the closest image to training set from test set (finding the position into the store) 


