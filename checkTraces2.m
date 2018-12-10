clear
clc
close all

%% data

dates = {'18' '19' '20'};
% dates = {'18'};

%% Loop

for DD = 1:3
    date_dir = ['2018.07.' dates{DD}];
    date =  ['2018-07-' dates{DD}];
    
    files = dir([pwd '\' date_dir '\Exp*']);
    n.exps = length(files);
    
    for EXP = 1:n.exps
        %% directories
        
        expNo = files(EXP).name;
        direct.data = [pwd '\' date_dir '\' expNo '\'];
        direct.results = [direct.data 'results\'];
        
        %% load data
        
        load([direct.results 'TRACES2.mat'])
        
        load([ direct.data 'MOVEDIRS_ ' date '_' expNo '.mat'])
        load([ direct.data 'Block_CaFrames_ ' date '_' expNo '.mat'])
        
        load([ direct.data 'MetaData_ ' date '_' expNo '.mat'])
        
        %% Metadata
        
        directions = COND.Movement_Direction;
        
        COND = unique(MOVEDIRS);
        n.cond = length(COND);
        
        n.frames = 29;
        n.trials = 800;
        n.trialsCond = n.trials/n.cond;
        n.neurons = size(TRACES,2);
        
        
        %% Z-score traces
        
        TRACES = TRACES - TRACES(1,:);
        M = mean(TRACES);
        STD = std(TRACES);
        
        TRACES_Z = (TRACES - M)./STD;
        
        h = figure;
        plot(TRACES_Z)
        title([date ' ' expNo])
        saveas(h, [direct.results 'allTraces.png'])
        saveas(h, [direct.results 'allTraces.fig'])
        
        %% Epoch Traces
        
        
        EPOCHS = NaN(n.neurons, n.frames, n.trialsCond, n.cond);
        for CC = 1:n.cond
            idx = find(MOVEDIRS == CC);
            
            start = Block_CaFrames(1,idx);
            stop = start + n.frames - 1;
            
            for ii = 1:n.trialsCond
                tmp = TRACES_Z(start(ii):stop(ii), :)';
                %        tmp = tmp - tmp(:,1);
                
                EPOCHS(:,:,ii, CC) = tmp;
                %        EPOCHS(:,:,ii, CC) = TRACES(start(ii):stop(ii), :)';
            end
        end
        
        %% Plot lines 1
        % figure;
        % neuron = 38;
        % cond = 3;
        %
        % dat = squeeze(EPOCHS(neuron, :, :, cond));
        % plot(dat)
        % title(['Neuron: ' num2str(neuron)])
        
        %% Plot lines
        figure;
        neuron = n.neurons;
        
        dat = squeeze(mean(EPOCHS(neuron, :, :, :),3));
        
        plot(dat)
        title(['Neuron: ' num2str(neuron)])
        
        %% Plot Result
        
        h = figure;
        for NN = 1:n.neurons
            subplot(ceil(n.neurons/8),8,NN)
            dat = squeeze(mean(mean(EPOCHS,2), 3));
            % dat = dat./max(dat,[],2);
            
            plot(directions, dat(NN,:)', 'k-', 'linewidth', 3)
            ylim([-0.5 0.5])
            title(NN)
        end
        % xlabel('Motion Direction')
        % ylabel('Relative Neuronal Response')
        
        suptitle([date ' ' expNo '  Tuning? by Neuron'])
        saveas(h, [direct.results 'Tuning.png'])
        
        %% Assess Tuning
        for NN = 1:n.neurons
            W_spike = dat(NN,:);
            theta_spike = W_spike*exp(directions * 1i);
            Tuning_Metric(NN) = abs(theta_spike);
        end
        
        figure;
        bar(sort(Tuning_Metric))
        
        
        %% centre around maximum neuron
        
        clear idx
        MP = ceil(n.cond/2); %mid point between 0 and 15;
        
        % Get data
        % dat = squeeze(mean(max(EPOCHS,[],2), 3));
        dat = squeeze(mean(mean(EPOCHS,2), 3));
        
        % dat = dat./mean(dat,2);
        
        % Make reshuffled data
        datnew = NaN(n.neurons,n.cond);
        for NN = 1:n.neurons
            
            % Get max point
            datuse = dat(NN,:) ;
            [~,CC] = max(datuse);
            
            
            if CC < MP
                idx.A1 = MP + (CC ); % where data comes from
                idx.A2 = MP - (CC -1); % where the data goes
                
                idx.B1 = MP + (CC +1); % where data comes from
                idx.B2 = MP - CC ; % where the data goes
            else % switching goes the other was around past the mid-point
                idx.A1 = CC - MP; % where data comes from
                idx.A2 = n.cond - (CC - MP) +1; % where the data goes
                
                idx.B1 = (CC - MP) +1 ; % where data comes from
                idx.B2 = n.cond - (CC - MP) ; % where the data goes
            end
            
            % shuffle around
            
            datnew(NN,idx.A2 : end) = datuse(1 : idx.A1);
            datnew(NN, 1 : idx.B2) = datuse(idx.B1 : end);
            
        end
        
        
        %% Summary Image
        n.directions=8;
        h = figure;
        
        subplot(2,2,1)
        imname = 'Image_0001_00012.png';
        IM = imread([direct.data imname]);
        imshow(IM(50:460,100:530,:))
        title([num2str(n.neurons) ' neurons identified'])
        
        subplot(2,2,2)
        plot(-3:4, datnew')
        xlabel('relative motion direction')
        ylabel('relative neuronal response')
        title('Corrected Response by Neuron')
        
        
        subplot(2,2,3)
        [~,ii]=max(dat, [],2);
        for DD = 1:n.directions
            n.neuronstuned(DD) = sum(ii==DD);
        end
        bar(n.neuronstuned)
        set(gca, 'xticklabel', directions)
        xlabel('Motion Direction (°)')
        ylabel('# of neurons')
        title('Number of neurons by direction')
        
        subplot(2,2,4)
        SNR = datnew(:,4) - mean(datnew(:,[1:3 5:8]),2);
        for DD = 1:n.directions
            M_ResponseSNR(DD) = mean(SNR(ii==DD));
        end
        bar( M_ResponseSNR)
        set(gca, 'xticklabel', directions)
        xlabel('Motion Direction (°)')
        ylabel('Mean Tuning Amp.')
        title('Mean Tuning Amplitude by direction')
        
        
        suptitle([date ' ' expNo])
        
        fig = gcf;
        fig.PaperUnits = 'points';
        fig.PaperPosition = [0 0 750 500];
        print([direct.results 'resultsummary.png'],'-dpng','-r0')
        
        
        %% Polar plot
        %
        % h = figure;
        %
        %
        % for neuron = 1:n.neurons
        %     subplot(8,5,neuron)
        %     % Get Data
        %     rho = squeeze(mean(mean(EPOCHS(neuron,:,:,:),2), 3));
        %
        %     rho = rho - min(rho);
        %     rho = rho./max(rho);
        %
        %     % Get axis
        %     theta = deg2rad(directions);
        %
        %     % - complete the circle
        %     theta = [theta; theta(1)];
        %     rho = [rho; rho(1)];
        %
        %     % Plot
        %     polarplot(theta, rho)
        %     hold on;
        %     axis('off')
        %     title(['Neuron: ' num2str(neuron)])
        % end
        %
        %
        %
        % saveas(h, 'Tuning_polar.png')
        
        %% SPLIT FIRST AND SECOND HALD
        %% centre around maximum neuron
        
        clear idx
        MP = ceil(n.cond/2); %mid point between 0 and 15;
        
        % Get data
        % dat = squeeze(mean(max(EPOCHS,[],2), 3));
        
        idx1 = round(n.trialsCond/2);
        dat = squeeze(mean(mean(EPOCHS,2), 3));
        dat1 = squeeze(mean(mean(EPOCHS(:,:,1:idx1,:),2), 3));
        dat2 = squeeze(mean(mean(EPOCHS(:,:,idx1:end,:),2), 3));
        
        % dat = dat./mean(dat,2);
        
        % Make reshuffled data
        datnew1 = NaN(n.neurons,n.cond);
        datnew2 = NaN(n.neurons,n.cond);
        for NN = 1:n.neurons
            
            % Get max point
            datuse = dat1(NN,:) ;
            [~,CC] = max(datuse);
            
            
            if CC < MP
                idx.A1 = MP + (CC ); % where data comes from
                idx.A2 = MP - (CC -1); % where the data goes
                
                idx.B1 = MP + (CC +1); % where data comes from
                idx.B2 = MP - CC ; % where the data goes
            else % switching goes the other was around past the mid-point
                idx.A1 = CC - MP; % where data comes from
                idx.A2 = n.cond - (CC - MP) +1; % where the data goes
                
                idx.B1 = (CC - MP) +1 ; % where data comes from
                idx.B2 = n.cond - (CC - MP) ; % where the data goes
            end
            
            % shuffle around
            
            datnew1(NN,idx.A2 : end) = datuse(1 : idx.A1);
            datnew1(NN, 1 : idx.B2) = datuse(idx.B1 : end);
            
            %% DAT2
            
            % Get max point
            datuse = dat2(NN,:) ;
            [~,CC] = max(datuse);
            
            
            if CC < MP
                idx.A1 = MP + (CC ); % where data comes from
                idx.A2 = MP - (CC -1); % where the data goes
                
                idx.B1 = MP + (CC +1); % where data comes from
                idx.B2 = MP - CC ; % where the data goes
            else % switching goes the other was around past the mid-point
                idx.A1 = CC - MP; % where data comes from
                idx.A2 = n.cond - (CC - MP) +1; % where the data goes
                
                idx.B1 = (CC - MP) +1 ; % where data comes from
                idx.B2 = n.cond - (CC - MP) ; % where the data goes
            end
            
            % shuffle around
            
            datnew2(NN,idx.A2 : end) = datuse(1 : idx.A1);
            datnew2(NN, 1 : idx.B2) = datuse(idx.B1 : end);
            
        end
        
        
        
        %% Summary Image
        n.directions=8;
        
        tmp = [datnew1(:); datnew2(:)];
        LIM= [min(tmp) max(tmp)];
        h = figure;
       
        subplot(2,2,1)
        plot(-3:4, datnew1')
        xlabel('relative motion direction')
        ylabel('relative neuronal response')
        title('Early')
        ylim(LIM)
        
        subplot(2,2,2)
        plot(-3:4, datnew2')
        xlabel('relative motion direction')
        ylabel('relative neuronal response')
        title('late')
        ylim(LIM)
%         
%         subplot(2,2,3)
%         [~,ii]=max(dat, [],2);
%         for DD = 1:n.directions
%             n.neuronstuned(DD) = sum(ii==DD);
%         end
%         bar(n.neuronstuned)
%         set(gca, 'xticklabel', directions)
%         xlabel('Motion Direction (°)')
%         ylabel('# of neurons')
%         title('Number of neurons by direction')
%         
        subplot(2,2,3)
        SNR = datnew1(:,4) - mean(datnew1(:,[1:3 5:8]),2);
        for DD = 1:n.directions
            M_ResponseSNR1(DD) = mean(SNR(ii==DD));
        end
        
         SNR = datnew2(:,4) - mean(datnew2(:,[1:3 5:8]),2);
        for DD = 1:n.directions
            M_ResponseSNR2(DD) = mean(SNR(ii==DD));
        end
        
        tmp = [M_ResponseSNR1(:); M_ResponseSNR2(:)];
        LIM = [0 max(tmp)+0.1];
        bar( M_ResponseSNR1)
        set(gca, 'xticklabel', directions)
        xlabel('Motion Direction (°)')
        ylabel('Mean Tuning Amp.')
        title('early')
        ylim(LIM)
        
        subplot(2,2,4)
       
        bar( M_ResponseSNR2)
        set(gca, 'xticklabel', directions)
        xlabel('Motion Direction (°)')
        ylabel('Mean Tuning Amp.')
        title('late')
        ylim(LIM)
        
        
        suptitle([date ' ' expNo])
        
        fig = gcf;
        fig.PaperUnits = 'points';
        fig.PaperPosition = [0 0 750 500];
        print([direct.results 'early vs late.png'],'-dpng','-r0')
        
        
        
        
    end
end