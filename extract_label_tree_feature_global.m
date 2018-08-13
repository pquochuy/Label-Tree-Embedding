function [feat_global, label_global] = extract_label_tree_feature_global(id, feat, label)
    ids = unique(id);
    feat_global = zeros(numel(ids),size(feat,2));
    label_global = zeros(numel(ids),1);
    for i = 1 : numel(ids)
        ind = (id == ids(i));
        feat_global(i,:) = sum(feat(ind,:))/sum(ind);
        label_global(i) = unique(label(ind));
    end
end