desc_name = 'dsift';

clearvars bof_test desc_test bof_l2lab

lasti=1;
for i = 1:length(data)
     images_descs = get_descriptors_files(data,i,file_ext,desc_name,'test');
     for j = 1:length(images_descs) 
        fname = fullfile('/media/andrea/Dati2/CV_Proj/handsonbow/dataset/validation/img',images_descs{j});
        fprintf('Loading %s \n',fname);
        tmp = load(fname,'-mat');
        tmp.desc.class=i;
        tmp.desc.imgfname=regexprep(fname,['.' desc_name],'.jpg');
        desc_test(lasti)=tmp.desc;
        desc_test(lasti).sift = single(desc_test(lasti).sift);
        lasti=lasti+1;
     end
end

for i=1:length(desc_test)
    
    dmat=eucliddist(desc_test(i).sift,VC);
    [mv, visword] = min(dmat, [], 2);
    
    % save feature labels
    desc_test(i).visword = visword;
    desc_test(i).quantdist = mv;
end


N = size(VC,1); % number of visual words

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


for i=1:length(desc_test)
    disp(desc_test(i).imgfname);
    desc_test(i).llc = max(LLC_coding_appr(VC,desc_test(i).sift));
    desc_test(i).llc=desc_test(i).llc/norm(desc_test(i).llc);
end


bof_train=cat(1,desc_train.bof);
bof_test=cat(1,desc_test.bof);
if do_svm_llc_linara_classification
    llc_train = cat(1,desc_train.llc);
    llc_test = cat(1,desc_test.llc);
end

% Construct label Concatenate bof-histograms into training and test matrices 
labels_train=cat(1,desc_train.class);
labels_test=cat(1,desc_test.class);


bof_l2dist=eucliddist(bof_test,bof_train);


% Nearest neighbor classification (1-NN) using L2 distance
[mv,mi] = min(bof_l2dist,[],2);
bof_l2lab = labels_train(mi);

method_name='NN L2';
acc=sum(bof_l2lab==labels_test)/length(labels_test);
fprintf('\n*** %s ***\nAccuracy = %1.4f%% (classification)\n',method_name,acc*100);