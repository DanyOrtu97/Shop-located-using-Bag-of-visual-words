function class = DetectPosition(input_image, folder, desc_name,nwords_codebook, desc_train)
    %This function allow us to find the class(the position in the store) of a given image
    
    extract_sift_features(folder,desc_name) 
    %load(fullfile('/media/andrea/Dati2/CV_Proj/handsonbow/dataset/','VC.mat'));
    
    sift_path = strcat(input_image(1:length(input_image)-4), '.', desc_name);
    tmp = load(fullfile(sift_path), '-mat');
    tmp.desc.imgfname=regexprep(input_image,['.' desc_name],'.jpg');
    desc_test=tmp.desc;
    desc_test.sift = single(desc_test.sift);
    
    
    
    
    dmat=eucliddist(desc_test.sift,VC);
    [mv, visword] = min(dmat, [], 2);
    
    % save feature labels
    desc_test.visword = visword;
    desc_test.quantdist = mv;
    
    H = histc(visword,[1:nwords_codebook]);
  
    % normalize bow-hist (L1 norm)

    H = H/sum(H);

  
    % save histograms
    desc_test.bof=H(:)';
    
    
    desc_test.llc = max(LLC_coding_appr(VC,desc_test.sift));
    desc_test.llc=desc_test.llc/norm(desc_test.llc);
    
    
    bof_train=cat(1,desc_train.bof);
    bof_test=cat(1,desc_test.bof);
    
    llc_train = cat(1,desc_train.llc);
    llc_test = cat(1,desc_test.llc);
    
    labels_train=cat(1,desc_train.class);
    %labels_train = string(labels_train);
    %labels_train = sort(labels_train);
    bof_l2dist=eucliddist(bof_test,bof_train);
  
    
    % Nearest neighbor classification (1-NN) using L2 distance
    [mv,mi] = min(bof_l2dist,[],2);
    bof_l2lab = labels_train(mi);
    
    class = classes(bof_l2lab);
    
    text = strcat("L'immagine appartiene al corridoio: ", string(class), "\n");
    fprintf(text);
end

