clc
clear
opts = get_opts();
all_data = h5read(fullfile(opts.dataset_path,'ground_truth','target_data.h5'),'/emb');
appear_score = all_data(1,:)';
% appear_score = (appear_score-0.49)/0.46;
motion_score = all_data(2,:)';
target = all_data(3,:)';
X = [ones(size(appear_score)),appear_score,motion_score,appear_score.*motion_score,appear_score.^2,motion_score.^2];
% X = [ones(size(appear_score)),appear_score,appear_score.^2];
b = regress(2*target-1,X)
y=X*b;
figure()
subplot(1,2,1)
title('softmax: 2*A-1')
miss1=sum(abs((appear_score-0.49)/0.46-2*target+1)>1)
histogram((appear_score-0.49)/0.46-2*target+1)
subplot(1,2,2)
title('2nd order A/M')
miss2=sum(abs(y-2*target+1)>1)
histogram(y-2*target+1)