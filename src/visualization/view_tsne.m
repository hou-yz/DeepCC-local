function view_tsne(distance,labels)
figure(6)
clf('reset');
hold on
%% Perform tSNE
uni_labels = unique(labels);
no_dims = 2;
perplexity = 30;
yd = tsne_d(distance, [], no_dims, perplexity);
%% Plot results
for i = 1:length(uni_labels)
    label = uni_labels(i);
    scatter(yd(labels==label, 1), yd(labels==label, 2))
end
hold off
title('tSNE embedding')
end

