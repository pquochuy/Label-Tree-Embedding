%% split dev data into train and evaluation sets
function [tr_X, tr_y, te_X, te_y] = split_data(X, y)

    % randomly divide 50/50 training/evaluation data
    tr_X = [];
    tr_y = [];
    te_X = [];
    te_y = [];

    class_id = unique(y);
    for cl = 1 : numel(class_id)
        ind = find(y == class_id(cl));
        % shuffle them up
        permlist = randperm(numel(ind));
        ind = ind(permlist);
        tr_X = [tr_X; X(ind(1:round(numel(ind)/2)),:)];
        tr_y = [tr_y; y(ind(1:round(numel(ind)/2)))];
        te_X = [te_X; X(ind(round(numel(ind)/2) + 1: end),:)];
        te_y = [te_y; y(ind(round(numel(ind)/2) + 1: end))];
    end
    
end
