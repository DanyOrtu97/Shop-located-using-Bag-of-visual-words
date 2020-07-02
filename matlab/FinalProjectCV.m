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
basepath = 'dataset/';
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
        %if ~exist(int2str(i), '/train_set/split_by_class_RGB/')
            mkdir(strcat(basepath, 'train_set/'), int2str(i));
        %end

        %if ~exist(int2str(i), '/test_set/split_by_class_RGB/')
            mkdir(strcat(basepath, 'test_set/'), int2str(i));
        %end
    end
end

if do_copyimages
    %Training set
    [M, N] = size(dataTrain);
    for i = 1 : M
        class = dataTrain(i,7);
        file_RGB = strcat(basepath, 'train_set/train_RGB/', dataTrain(i,1));
        dest = strcat(basepath, 'train_set/', num2str(cell2mat(class)));
        copyfile(string(file_RGB), dest);
    end
    
    %Test set
    [M, N] = size(dataTest);
    for i = 1 : M
        class = dataTest(i,7);
        file_RGB = strcat(basepath, 'test_set/test_RGB/', dataTest(i,1));
        dest = strcat(basepath, 'test_set/', num2str(cell2mat(class)));
        copyfile(string(file_RGB), dest);
    end
end

% Create a new dataset split for train
file_split = 'split.mat';
if do_split_sets    
    data = create_dataset_split_structure(basepath, file_ext);
    save(fullfile(basepath, '/train_set/', file_split), 'data');
else
    load(fullfile(basepath, '/train_set/', file_split));
end

classes = {data.classname}; % create cell array of class name strings

% Extract SIFT features fon training and test images
if do_feat_extraction  
    extract_sift_features(fullfile(basepath, '/train_set/'), desc_name);   
end


% Convert images of the training set into BoVW representation


% Induction of the classifier


% Knn to find the closest image to training set from test set (finding the position into the store) 


