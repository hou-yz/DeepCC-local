%% Fetching data
opts = get_opts_aic;
if ~exist(fullfile(opts.dataset_path,'ground_truth/train.mat'),'file')
    fprintf('Downloading ground truth...\n');
    url = 'https://drive.google.com/uc?export=download&id=1_fWMD3kMawp9hOaHnkpCKIYgpt9vufsQ';
    if ~exist(fullfile(opts.dataset_path,'ground_truth'),'dir'), mkdir(fullfile(opts.dataset_path,'ground_truth')); end
    filename = fullfile(opts.dataset_path,'ground_truth/train.mat');
    if exist('websave','builtin')
      outfilename = websave(filename,url); % exists from MATLAB 2014b
    else
      outfilename = urlwrite(url, filename);
    end
end
if ~exist('experiments/aic_demo','dir') || ~exist('experiments/aic_demo/train.txt','file') %|| ~exist('experiments/aic_demo/test.txt','file')
    fprintf('Downloading baseline tracker output...\n');
    url = 'https://drive.google.com/uc?export=download&id=1qQ1PJXyrdKb8kU0NWlZtAlWfB4oFv__O';
    if ~exist('experiments/aic_demo','dir'), mkdir('experiments/aic_demo'); end
    filename = 'experiments/aic_demo/baseline.zip';
    if exist('websave','builtin')
      outfilename = websave(filename,url); % exists from MATLAB 2014b
    else
      outfilename = urlwrite(url, filename);
    end
    unzip(outfilename,'experiments/aic_demo/');
    delete(filename);
    % Convert to motchallenge format: Frame, ID, left, top, right, bottom,
    % worldX, worldY
    output_train = dlmread('experiments/aic_demo/train.txt');
    %output_test = dlmread('experiments/aic_demo/test.txt');
    
    for cam = 1:40
        filter_train = output_train(:,1) == cam;
        data_train = output_train(filter_train,:);
        data_train = data_train(:,2:end);
        data_train(:,[1 2]) = data_train(:,[2 1]);
        if ~isempty(data_train)
            dlmwrite(sprintf('experiments/aic_demo/c%03d_train.txt', cam), data_train, 'delimiter', ',', 'precision', 6);
        end

%         filter_test = output_test(:,1) == cam;
%         data_test = output_test(filter_test,:);
%         data_test = data_test(:,2:end);
%         data_test(:,[1 2]) = data_test(:,[2 1]);
%         if ~isempty(data_test)
%             dlmwrite(sprintf('experiments/aic_demo/c%03d_test.txt', cam), data_test, 'delimiter', ',', 'precision', 6);
%         end
    end
end

%% Evaluation
[allMets, metsBenchmark, metsMultiCam] = evaluateTracking('AIC19-train.txt', 'experiments/aic_demo/', 'gt/AIC19', 'AIC19',opts.dataset_path);
