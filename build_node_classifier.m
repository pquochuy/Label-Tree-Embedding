function build_node_classifier(skeleton_path, tree_path, tr_data,mtree,mtry)

    nodes = dir([skeleton_path,'*.mat']);
    num_node = numel(nodes);

    for i = 1 : num_node
	
	    %nodes = dir([skeleton_path,'*.mat']);
		%num_node = numel(nodes);
	
        if(exist([tree_path,'node_',num2str(i),'.mat'],'file'))
            continue;
        end
        load([skeleton_path,'node_',num2str(i),'.mat']);
        if(cur_node.isleaf == 1)
            save([tree_path,'node_',...
                num2str(i),'.mat'],'cur_node','-v7.3');
            disp(['CONTINUE: ', tree_path, 'node_',num2str(i)]);
            continue;
        end
        disp([tree_path, 'node_',num2str(i)]);
        tr_X = [];
        tr_y = [];
        ind = ismember(tr_data.y,cur_node.left_labels);
        tr_X = [tr_X; tr_data.X(ind,:)];
        tr_y = [tr_y; zeros(sum(ind),1)];
        ind = ismember(tr_data.y,cur_node.right_labels);
        tr_X = [tr_X; tr_data.X(ind,:)];
        tr_y = [tr_y; ones(sum(ind),1)];

        N = length(tr_y);
        classwt = N./histc(tr_y,unique(tr_y));
        classwt = classwt';
        clear extra_options;
        extra_options.do_trace = 0;
        extra_options.nodesize = 10; % minimum samples to stop splitting --> reduce memory
        extra_options.classwt = classwt;
        model = classRF_train(tr_X,tr_y, mtree,mtry,extra_options);
        cur_node.model = model;
        save([tree_path,'node_',...
                num2str(i),'.mat'],'cur_node','-v7.3');
    end
end
