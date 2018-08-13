function [feat] = extract_label_tree_feature(tree_path, X)

    nodes = dir([tree_path,'*.mat']);
    num_node = numel(nodes);
    
    feat = [];
    
    for i = 1 : num_node
        load([tree_path,'node_',num2str(i),'.mat']);
        if(cur_node.isleaf == 1)
            continue;
        end
        [~,prob] = classRF_predict(X,cur_node.model);
        prob = prob/cur_node.model.ntree;
        feat = [feat, prob];
    end
end