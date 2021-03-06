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

% Extraction of data from text file
path = '/media/andrea/Dati2/CV_Proj/handsonbow/dataset/train_set/train_set.txt';
base_path = '/media/andrea/Dati2/CV_Proj/handsonbow/dataset/train_set/';
data = readtable(path,'Delimiter', ' ', 'HeaderLines', 0, 'ReadVariableNames',true );
data = data(:,1:7);

testMatrix = table2cell(data);
[M, N] = size(testMatrix);

for i = 1:16
    mkdir('/media/andrea/Dati2/CV_Proj/handsonbow/dataset/train_set/split_by_class', int2str(i))
end

for i = 1 : M
    class = testMatrix(i,7);
    file_RGB = strcat('/media/andrea/Dati2/CV_Proj/handsonbow/dataset/train_set/train_RGB/', testMatrix(i,1));
    dest = strcat('/media/andrea/Dati2/CV_Proj/handsonbow/dataset/train_set/split_by_class/', num2str(cell2mat(class)));
    copyfile(string(file_RGB), dest) 
end


% Inizializations
%desc_name = 'sift';
desc_name = 'dsift';
%desc_name = 'msdsift';


% FLAGS
do_feat_extraction = 0;
do_split_sets = 0;

do_form_codebook = 1;
do_feat_quantization = 1;

do_L2_NN_classification = 1;

visualize_confmat = 0;
visualize_res = 0;
have_screen = ~isempty(getenv('DISPLAY'));


% PATHS
basepath = '/media/andrea/Dati2/CV_Proj/handsonbow/';
wdir = '/media/andrea/Dati2/CV_Proj/handsonbow/';
libsvmpath = [ wdir, fullfile('lib','libsvm-3.11','matlab')];
addpath(libsvmpath)

% BOW PARAMETERS
max_km_iters = 50; % maximum number of iterations for k-means
nfeat_codebook = 60000; % number of descriptors used by k-means for the codebook generation
norm_bof_hist = 1;

% number of codewords (i.e. K for the k-means algorithm)
nwords_codebook = 500;

% image file extension
file_ext='jpg';

% Create a new dataset split
file_split = 'split.mat';
file_split30 = 'split30.mat';
file_split70 = 'split70.mat';

if do_split_sets    
    data = create_dataset_store(fullfile(basepath, 'dataset'), 1, file_ext,70)
    save(fullfile(basepath,'dataset',dataset_dir,file_split70),'data');
else
    load(fullfile(basepath,'dataset',file_split));
end
classes = {data.classname}; % create cell array of class name strings

% Extract SIFT features fon training and test images
if do_feat_extraction   
    extract_sift_features(fullfile(basepath, 'dataset/train_set/split_by_class'),desc_name)    
end



%test dataset validation
data = create_dataset_detect(fullfile(basepath, 'dataset'), file_ext,100)



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

% Concatenate bof-histograms into training and test matrices 
bof_train=cat(1,desc_train.bof);
bof_test=cat(1,desc_test.bof);
if do_svm_llc_linara_classification
    llc_train = cat(1,desc_train.llc);
    llc_test = cat(1,desc_test.llc);
end

% Construct label Concatenate bof-histograms into training and test matrices 
labels_train=cat(1,desc_train.class);
labels_test=cat(1,desc_test.class);


% NN classification
tic
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
 
time = toc;


%Find the position into the store 
input_image = '/media/andrea/Dati2/CV_Proj/handsonbow/dataset/validation/img/3L024980.jpg';
folder = '/media/andrea/Dati2/CV_Proj/handsonbow/dataset/validation/';
DetectPosition(input_image, desc_name,nwords_codebook, desc_train);




