
figure(5)
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.1, 1, 0.7]);
clf('reset');

uni_labels = unique(in_window_pids);
no_dims = 2;
if mct
    perplexity = 8;
else
    perplexity = 18;
end

map_size=100;

%% Perform tSNE
% yd = tsne_d(exp(-distanceG), [], no_dims, perplexity);
% yd(:,1) = yd(:,1)*map_size/(mean(yd(:,1).^2)^0.5);
% yd(:,2) = yd(:,2)*map_size/(mean(yd(:,2).^2)^0.5);
% ydG=yd;

yd=ydG;
% axis([-map_size(2),map_size(2),-map_size(1),map_size(1)]*2)
%% Plot results
subplot(1,2,1)

view_tsne_2_function

title(['tSNE embedding',newline,'traditional metric learning'])

%% Perform tSNE
% yd = tsne_d(exp(-distanceL), [], no_dims, perplexity);
% yd(:,1) = yd(:,1)*map_size/(mean(yd(:,1).^2)^0.5);
% yd(:,2) = yd(:,2)*map_size/(mean(yd(:,2).^2)^0.5);
% ydL=yd;

yd=ydL;
% axis([-map_size(2),map_size(2),-map_size(1),map_size(1)]*2)
%% Plot results
subplot(1,2,2)

view_tsne_2_function

title(['tSNE embedding',newline,'TLML'])


