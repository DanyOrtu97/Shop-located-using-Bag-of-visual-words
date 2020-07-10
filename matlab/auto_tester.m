% This code help us to automatically compute the accuracy and execution time from different inputs 

dataset_dir='';

%desc_name = 'sift';
%desc_name = 'dsift';
desc_name = 'msdsift';

% FLAGS
visualize_confmat = 0;
visualize_res = 0;
have_screen = ~isempty(getenv('DISPLAY'));

% PATHS
basepath = pwd;
wdir = pwd;
libsvmpath = [ wdir, fullfile('/lib','libsvm-3.11','matlab')];
addpath(libsvmpath)
fileID = fopen('C:\Users\Daniele\Desktop\Daniele\1. Università\2. Magistrale\1° ANNO 2° SEMESTRE\Computer Vision\Progetto\CV_FinalProject-master/results.txt','w');

% BOW PARAMETERS
max_km_iters = 50; % maximum number of iterations for k-means
nfeat_codebook = 60000; % number of descriptors used by k-means for the codebook generation
norm_bof_hist = 1;

% image file extension
file_ext='jpg';

% Inizializations/Test for the printf
test_perc = 30;
test_nw = 100;
test_time = 5.3;

% Create a new dataset split
file_split = 'split.mat';
file_split30 = 'split30.mat';
file_split70 = 'split70.mat';

percentages = [30 70 100];
nwords = [10 100 200 300 400 500];
for i1 = 1:3
    perc = percentages(i1);
    for j1 = 1:6
        nwords_codebook = nwords(j1);
        if perc == 30
            load(fullfile(basepath,'dataset',dataset_dir,file_split30));
        end
        if perc == 70
            load(fullfile(basepath,'dataset',dataset_dir,file_split70));
        end
        if perc == 100
            load(fullfile(basepath,'dataset',dataset_dir,file_split));
        end
        classes = {data.classname}; % create cell array of class name strings
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
        
        
        
        
        bof_train=cat(1,desc_train.bof);
        bof_test=cat(1,desc_test.bof);
        llc_train = cat(1,desc_train.llc);
        llc_test = cat(1,desc_test.llc);
        labels_train=cat(1,desc_train.class);
        labels_test=cat(1,desc_test.class);
        
        tic
        % Compute L2 distance between BOFs of test and training images
        bof_l2dist=eucliddist(bof_test,bof_train);
        
        
        % Nearest neighbor classification (1-NN) using L2 distance
        [mv,mi] = min(bof_l2dist,[],2);
        bof_l2lab = labels_train(mi);
        
        method_name='NN L2';
        acc=sum(bof_l2lab==labels_test)/length(labels_test);
        fprintf('\n*** %s ***\nAccuracy = %1.4f%% (classification)\n',method_name,acc*100);
        
        % Compute classification accuracy
        overall_accuracy = compute_accuracy(data,labels_test,bof_l2lab,classes,method_name,desc_test,...
            visualize_confmat & have_screen,...
            visualize_res & have_screen);
        
        time = toc;
        
        % Print the results on text file
        test_text = strcat( "\ndsift-> nwords: ", string(nwords_codebook), "   train_percentage: ", string(perc), "   time: ", string(time), "  accuracy: ", string(overall_accuracy));
        fprintf(fileID, test_text);
        
        % Clear the bigger vars in order to avoid ram saturation
        clearvars desc_test desc_class desc_train
    end
end

