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
basepath = '../dataset/';
libsvmpath = [fullfile('lib','libsvm-3.11','matlab')];
addpath(libsvmpath);
limitDatasetSize = 10;

% Do or not some operations
importTrainingSetData = 1;
importTestSetData = 0;

% Image file extension
file_ext='jpg';

nameColumns = {'rgbFile', 'depthFile', 'x', 'y', 'u', 'v', 'class'};
dataTxt = readtable(fullfile('dataset', 'train_set', 'train_set.txt'));
dataTxt = dataTxt(:,1:7);
% classname = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16'};
% dim = 13360;

% creation od a split.mat file
% save split.mat dim classname;

% Acquisition of the test and training set
if importTrainingSetData
    % load(fullfile('dataset', 'train_set', 'train_RGB'));

    % TrainingSetData = ...
end

if importTestSetData
    % TestSetData = ...
end




% Convert images of the training set into BoVW representation


% Induction of the classifier


% Knn to find the closest image to training set from test set (finding the position into the store) 


