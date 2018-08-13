%% construct label tree
function build_tree_skeleton(skeleton_path,tr_X, tr_y, te_X, te_y, mtree, mtry)

    % root node
    node_lables = {unique([tr_y; te_y])};
    node_status = [-1];

    empty_node_index = find(node_status == -1);

    while(~isempty(empty_node_index))
        clear cur_node;
        cur_node.index = empty_node_index(1);
        % extract data according to current class labels
        labels = node_lables{cur_node.index};
        if(numel(unique(labels)) == 1)
            cur_node.isleaf = 1;
            cur_node.labels = labels;
            cur_node.model = [];
            cur_node.left_labels = [];
            cur_node.right_labels = [];
            save([skeleton_path,'node_',...
            num2str(cur_node.index),'.mat'],'cur_node','-v7.3');
            % set status to complete
            node_status(cur_node.index) = 1;
        else
            cur_node.isleaf = 0;

            ind = ismember(tr_y, labels);
            tr_X_ = tr_X(ind,:);
            tr_y_ = tr_y(ind);
            ind = ismember(te_y, labels);
            te_X_ = te_X(ind,:);
            te_y_ = te_y(ind);

            % train RF classification model
            N = length(tr_y_);
            classwt = N./histc(tr_y_,labels);
            classwt = classwt';

            clear extra_options;
            extra_options.do_trace = 0;
            extra_options.nodesize = 10; % minimum samples to stop splitting --> reduce memory
            extra_options.classwt = classwt;
            model = classRF_train(tr_X_,tr_y_, mtree,mtry,extra_options);
            %cur_node.model = model;

            % test RF classification model
            [~,prob] = classRF_predict(te_X_,model);
            prob = prob/model.ntree;

            % calculate confusion matrix based on prob
            C_matrix = zeros(numel(labels),numel(labels));
            for i = 1 : numel(labels)
                ind = (te_y_ == labels(i));
                prob_i = prob(ind,:);
                C_matrix(i,:) = mean(prob_i);
            end
            % make confusion matrix symmetric
            C_matrix = (C_matrix + C_matrix')/2;

            % spectral clustering to from 2 label clusters
            init_spectral;
            number_of_clusters = 2;
            %assignment=cluster_algo(C_matrix,number_of_clusters);
            assignment = cluster_spectral_general(C_matrix,number_of_clusters,'ang_gen','kmeans');

            % divide to left and right nodes
            left_labels = labels(assignment == 1);
            right_labels = labels(assignment == 2);
            node_lables = [node_lables; left_labels; right_labels];
            node_status = [node_status; -1; -1];

            cur_node.labels = labels;
            cur_node.left_labels = left_labels;
            cur_node.right_labels = right_labels;

            save([skeleton_path,'node_',...
                num2str(cur_node.index),'.mat'],'cur_node','-v7.3');
            % set status to complete
            node_status(cur_node.index) = 1;
        end
        empty_node_index = find(node_status == -1);
    end
end
end
