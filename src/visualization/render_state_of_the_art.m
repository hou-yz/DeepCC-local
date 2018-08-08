% Shows state of the art on ID precision/recall plot
% Scores need to be updated from MOTChallenge. To be used as reference.

opts = get_opts();

close all;
difficulty = {'Hard', 'Easy'};
scenario = {'Multi', 'Single'};

% Create folder
folder = 'state-of-the-art';
mkdir([opts.experiment_root, filesep, opts.experiment_name, filesep, folder]);

% Difficulty
for ii = 1:2
    data = dlmread(sprintf('src/visualization/data/duke_%s_scores.txt',lower(difficulty{ii})));
    
    % Scenario
    for jj = 1:2
        P = data(:,jj*2-1)/100;
        R = data(:,jj*2)/100;
        figure;
        clf('reset');
        axis equal;
        axis([0 1 0 1]);
        % F-score isocontours
        for f = 0.2:0.1:0.9
            p = linspace(0.1,1,1000);
            r = (p*f)./(2*p-f);
            r(r<0) = 1.01;
            r(r>1) = 1.01;
            hold on; plot(p,r,'g-');
        end
        
        hold on; scatter(P,R,'filled');
        xlabel('ID Precision');
        ylabel('ID Recall');
        title(sprintf('DukeMTMC %s-Camera %s',scenario{jj},difficulty{ii}));
        set(gca, 'Color', 'none');
        figure_name = sprintf('duke_%s_%s.pdf',scenario{jj},difficulty{ii});
        figure_name = fullfile(opts.experiment_root, opts.experiment_name, folder, figure_name);
        export_fig('-transparent', figure_name);
        
    end
end