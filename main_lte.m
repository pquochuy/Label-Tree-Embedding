clear all
close all
clc

addpath('LTE');
addpath(genpath('RF'));
addpath(genpath('SpectralLib'));

mtree = 200;
mtry = 8;
nclass = 10;

% path where the mat files are saved
mat_path = './mat_files/';

% load training data
load([mat_path, 'train_data.mat']);
% load test data
load([mat_path, 'test_data.mat']);

% normalization
meanX = mean(train_data);
stdX = std(train_data);
train_data = train_data - repmat(meanX,size(train_data,1),1);
train_data = train_data ./ repmat(stdX,size(train_data,1),1);
test_data = test_data - repmat(meanX,size(test_data,1),1);
test_data = test_data ./ repmat(stdX,size(test_data,1),1);

% split to train and avaluation sets
if(~exist([mat_path, 'split_data.mat'],'file'))
    [tr_X, tr_y, te_X, te_y] = split_data(train_data, train_label);
    save([mat_path, 'split_data.mat'],'tr_X', 'tr_y', 'te_X', 'te_y');
end

% load splitted data
load([mat_path, 'split_data.mat']);
% path where to save tree nodes
skeleton_path = './tree_skeleton/'; 
if(~exist(skeleton_path,'dir'))
    mkdir(skeleton_path);
end
% learn the label tree
build_tree_skeleton(skeleton_path,tr_X, tr_y, te_X, te_y, mtree, mtry);
clear tr_X tr_y te_X te_y

% the number of cross-validataion folds on training data
Nfold = 10;
if(~exist([mat_path, 'fold_division.mat'],'file'))
    fold_division = cell(Nfold,1);
    for cl = [1 : nclass]
        ind = find(train_label == cl);
        permlist = randperm(numel(ind));
        ind = ind(permlist); % shuffle them up
        N = length(ind);
        for cv = 1 : Nfold
            ind_cv = ind([floor((cv-1)*N/10)+1 : floor(cv*N/10)]');
            fold_division{cv} = [fold_division{cv}; ind_cv];
        end
    end
    save([mat_path, 'fold_division.mat'],'fold_division');
end

% create the tree (mainly rebuild the node classifiers) from the skeleton
% do it ten folds of the training data to extract feature for the training data
for fold = 1 : Nfold
    % path where to save the tree nodes
    tree_path = ['./mat_files/cv', num2str(fold),'/'];
    if(~exist(tree_path,'dir'))
        mkdir(tree_path);
    end
    % load fold division
    load([mat_path, 'fold_division.mat']);
    ind = (~ismember([1:size(train_data,1)]', fold_division{fold}));
    traindata.X = train_data(ind,:); 
    traindata.y = train_label(ind);
    % build the classifiers at the tree nodes with the whole training data
    build_node_classifier(skeleton_path, tree_path, traindata,mtree,mtry);
end

% extract feature for the training data
train_feat = zeros(length(train_label),(nclass - 1)*2);
for fold = 1 : Nfold
    tree_path = ['./mat_files/cv', num2str(fold),'/'];
    X = extract_label_tree_feature(tree_path, train_data(fold_division{fold},:));
    train_feat(fold_division{fold},:) = X;
end
save([mat_path,'train_feat.mat'],'train_feat','train_label');  


% create the tree (mainly rebuild the node classifiers) from the skeleton
% this is for extract feature for the test data
tree_path = ['./mat_files/cv/'];
if(~exist(tree_path,'dir'))
    mkdir(tree_path);
end
traindata.X = train_data;
traindata.y = train_label;
build_node_classifier(skeleton_path, tree_path, traindata,mtree,mtry);
% extract LTE features for test data
test_feat = extract_label_tree_feature(tree_path, test_data);
save([mat_path,'test_feat.mat'],'test_feat','test_label');

