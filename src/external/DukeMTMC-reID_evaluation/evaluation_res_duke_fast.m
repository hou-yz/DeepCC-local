function evaluate_res_duke_fast(opts)

mkdir src/external/DukeMTMC-reID_evaluation/data
rank_size = 4000;

%% add necessary paths
query_dir = fullfile(opts.dataset_path, 'DukeMTMC-reID/query/');% query directory
test_dir = fullfile(opts.dataset_path, 'DukeMTMC-reID/bounding_box_test/');% src/external/DukeMTMC-reID_evaluation/database directory

%% calculate query features
Hist_query = h5read(fullfile('src/triplet-reid/', opts.net.experiment_root, 'duke_query_embeddings.h5'), '/emb');
nQuery = size(Hist_query, 2);

%% calculate src/external/DukeMTMC-reID_evaluation/database features
Hist_test = h5read(fullfile('src/triplet-reid/', opts.net.experiment_root, 'duke_test_embeddings.h5'), '/emb');
nTest = size(Hist_test, 2);

%% calculate the ID and camera for src/external/DukeMTMC-reID_evaluation/database images
test_files = dir([test_dir '*.jpg']);
testID = zeros(length(test_files), 1);
testCAM = zeros(length(test_files), 1);
if ~exist('src/external/DukeMTMC-reID_evaluation/data/testID_duke.mat')
    for n = 1:length(test_files)
        img_name = test_files(n).name;
        if strcmp(img_name(1), '-') % junk images
            testID(n) = -1;
            testCAM(n) = str2num(img_name(5));
        else
            %img_name
            testID(n) = str2num(img_name(1:4));
            testCAM(n) = str2num(img_name(7));
        end
    end
    save('src/external/DukeMTMC-reID_evaluation/data/testID_duke.mat', 'testID');
    save('src/external/DukeMTMC-reID_evaluation/data/testCAM_duke.mat', 'testCAM');
else
    testID = importdata('src/external/DukeMTMC-reID_evaluation/data/testID_duke.mat');
    testCAM = importdata('src/external/DukeMTMC-reID_evaluation/data/testCAM_duke.mat');    
end

%% calculate the ID and camera for query images
query_files = dir([query_dir '*.jpg']);
queryID = zeros(length(query_files), 1);
queryCAM = zeros(length(query_files), 1);
if ~exist('src/external/DukeMTMC-reID_evaluation/data/queryID_duke.mat')
    for n = 1:length(query_files)
        img_name = query_files(n).name;
        if strcmp(img_name(1), '-') % junk images
            queryID(n) = -1;
            queryCAM(n) = str2num(img_name(5));
        else
            queryID(n) = str2num(img_name(1:4));
            queryCAM(n) = str2num(img_name(7));
        end
    end
    save('src/external/DukeMTMC-reID_evaluation/data/queryID_duke.mat', 'queryID');
    save('src/external/DukeMTMC-reID_evaluation/data/queryCAM_duke.mat', 'queryCAM');
else
    queryID = importdata('src/external/DukeMTMC-reID_evaluation/data/queryID_duke.mat');
    queryCAM = importdata('src/external/DukeMTMC-reID_evaluation/data/queryCAM_duke.mat');    
end

%% search the src/external/DukeMTMC-reID_evaluation/database and calcuate re-id accuracy
ap = zeros(nQuery, 1); % average precision
ap_max_rerank  = zeros(nQuery, 1); % average precision with MultiQ_max + re-ranking 
ap_pairwise = zeros(nQuery, 6); % pairwise average precision with single query (see Fig. 7 in the paper)

CMC = zeros(nQuery, rank_size);
CMC_max_rerank = zeros(nQuery, rank_size);

r1 = 0; % rank 1 precision with single query
r1_max_rerank = 0; % rank 1 precision with MultiQ_max + re-ranking
r1_pairwise = zeros(nQuery, 6);% pairwise rank 1 precision with single query (see Fig. 7 in the paper)

dist = sqdist(Hist_test, Hist_query); % distance calculate with single query. Note that Euclidean distance is equivalent to cosine distance if vectors are l2-normalized
%dist_cos_max = (2-dist_max)./2; % cosine distance with MultiQ_max, used for re-ranking

knn = 1; % number of expanded queries. knn = 1 yields best result

for k = 1:nQuery
    % load ground truth for each query (good and junk)
    good_index = intersect(find(testID == queryID(k)), find(testCAM ~= queryCAM(k)))';% images with the same ID but different camera from the query
    junk_index1 = find(testID == -1);% images neither good nor bad in terms of bbox quality
    junk_index2 = intersect(find(testID == queryID(k)), find(testCAM == queryCAM(k))); % images with the same ID and the same camera as the query
    junk_index = [junk_index1; junk_index2]';
    score = dist(:, k);
    
    % sort src/external/DukeMTMC-reID_evaluation/database images according Euclidean distance
    [~, index] = sort(score, 'ascend');  % single query
    
    % re-rank  select rank_size=1000 index
    index = index(1:rank_size);    
    
    [ap(k), CMC(k, :)] = compute_AP(good_index, junk_index, index);% compute AP for single query
    fprintf('%d::%f\n',k,ap(k));
end
CMC = mean(CMC);
%% print result
fprintf('single query:                                    mAP = %f, r1 precision = %f\r\n', mean(ap), CMC(1));
%[ap_CM, r1_CM] = draw_confusion_matrix(ap_pairwise, r1_pairwise, queryCam);
%fprintf('average of confusion matrix with single query:  mAP = %f, r1 precision = %f\r\n', (sum(ap_CM(:))-sum(diag(ap_CM)))/30, (sum(r1_CM(:))-sum(diag(r1_CM)))/30);

%% plot CMC curves
figure;
s = 50;
CMC_curve = CMC ;
plot(1:s, CMC_curve(:, 1:s));
