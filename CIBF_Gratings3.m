clear
clc
close all

%% Session Settings

subject_ID = 'M0'; %To keep track of the subject across sessions
angles.train_idx = 1; %1 = -45°, 2 = 45°!

subject_SesionNotes = ''; % Part of saved name, things like Awake, Isoflourine etc - 

%% Screen Settings

monitor.use = 1;

monitor.width = 530; % mm
monitor.viewdist = 250; % mm

%% Grating Settings

grating.spatial_freq_DVA = 0.04;%; %cycles per pixel
grating.Speed_cylces = 2; % Hz (cycles per second)

ellipse.fadeoff = 50;
ellipse.diameter = [1820 980];

%% Timing settings

s.gratingMove = 1; %seconds

%% Setup Directories and filename

direct.Data_All = ['..\Data\'];
direct.Data = [direct.Data_All subject_ID '\'];
if ~exist(direct.Data)
    mkdir(direct.Data)
end


fname.date = strrep( date, '-', '.' );
fname.start_clock = clock;

fname.session = [ fname.date '_' num2str(fname.start_clock(4)) '.' num2str(fname.start_clock(5))];
fname.session = [ subject_ID '_' subject_SesionNotes '_' fname.session ];

%% Motion settings

angles.train = [-45 45]; % the two possible training angles;
angles.add_offset = [-60 60]; % the two possible offset angles to each training angles
angles.movedirs = [-1 1];

for ii = 1:2 % get the additional angles for train and untrain (+/- 60°)
    angles.add(ii,:) = mod(angles.train + angles.add_offset(ii) +180, 360)-180';
end

% Get movedirs for each angle (+/-90°)
movedirs.train(:,1) = mod(angles.train - 90 + 180, 360) -180;
movedirs.train(:,2) = mod(angles.train + 90 + 180, 360) -180;

movedirs.add(:,:,1) = mod(angles.add - 90 + 180, 360) -180;
movedirs.add(:,:,2) = mod(angles.add + 90 + 180, 360) -180;

%% Get average movement directions
for TRii = 1:2 % for -45 and 45 orientationdegrees
    for TRjj = 1:2 %dirs available for train grating [+/- 90°]
        for UTRii = 1:2 % for add -60 and +60 orientationdegrees
            for UTRjj = 1:2 %dirs available for add grating [+/- 90°]
                
                tmpangs.train = squeeze(movedirs.train(TRii,TRjj));
                tmpangs.add = movedirs.add(UTRii,TRii,UTRjj) ; 
                
                % obtain circular mean
                r = sum(exp(1i*deg2rad([tmpangs.train tmpangs.add])),2);
                AveDir(UTRjj, UTRii, TRjj,TRii) = rad2deg(angle(r));
                
            end
        end
    end
end

%% Actual motion directions
tmp1 = AveDir(:,:,:,1);
tmp2 = AveDir(:,:,:,2);

% movedirs2 = [tmp1(:) tmp2(:)];
movedirs2 = [unique(tmp1(:)) unique(tmp2(:))];
movedirs2_rad = deg2rad(movedirs2);

figure; 
for ii = 1:length(movedirs2_rad)
    polarplot([0 movedirs2_rad(ii,1)],[0 1], 'r', 'lineWidth', 2)
    hold on;
    polarplot([0 movedirs2_rad(ii,2)],[0 0.9], 'b')
    
end

%% Counterbalancing - setup DATA

% Condition settings
n.movedirs = length(movedirs2);
n.trainAngs = 2;
n.GlobalLocalCond = 2;

% Timing and trial numbers
s.exp = 1800; %seconds - half an hour

tmp = s.exp./s.gratingMove;
n.trials_movedir = floor(tmp/(n.movedirs*n.GlobalLocalCond));


n.trials = n.trials_movedir*n.movedirs*n.GlobalLocalCond;
n.trials_GlobalLocal = n.trials/n.GlobalLocalCond;

%Display results
disp('#################')
disp(['Experiment duration: ' num2str((n.trials*s.gratingMove)/60) ' mins'])
disp([num2str(n.trials_movedir) ' trials per direction and condition'])
disp('#################')


% DATA
str.DATA = {
    'Trial'
    'Block'
    
    'globallocal'
    
    'Movement_Direction'
    
    
    'TrainAngle_idx'
    'TrainAngle_movedir_idx'
    
    'AdditionalAngle_idx'
    'AdditionalAngle_movedir_idx'
    
    'TrainAngle'
    'TrainAngle_movedir'
    
    'AdditionalAngle'
    'AdditionalAngle_movedir'
    
    'starting_phase'};

n.D = length(str.DATA);

for ii = 1:n.D
   D.(str.DATA{ii}) = ii; 
end

DATA = NaN(n.trials, n.D);

% Trials
DATA(:,D.Trial) = 1:n.trials;

% randomise global vs local. 
GL = [];
for ii = 1:n.trainAngs
    GL = [GL; ones(n.trials_GlobalLocal,1)*ii];
end

GL = GL(randperm(n.trials));
DATA(:,D.globallocal) = GL;

% train ang present

traindirs = [];
for ii = 1:n.trainAngs
    traindirs = [traindirs; ones(n.trials_GlobalLocal/n.trainAngs,1)*ii];
end

traindirs = traindirs(randperm(n.trials_GlobalLocal));
DATA(DATA(:,D.globallocal)==1,D.TrainAngle_idx) = traindirs;

% Movement Directions
for GL = 1:n.GlobalLocalCond
    traindirs = [];
    for ii = 1:n.movedirs
        traindirs = [traindirs; ones(n.trials_movedir/(n.GlobalLocalCond/GL),1)*ii];
    end
    
    % Designate movement directions
    if GL == 1
        for ii = 1:n.trainAngs
            DATA(DATA(:,D.TrainAngle_idx)==ii, D.Movement_Direction) = movedirs2(traindirs(randperm(n.trials/4)),ii);
        end
    else
        DATA(DATA(:,D.globallocal)==2, D.Movement_Direction) = movedirs2(traindirs(randperm(n.trials/2)),1);
    end
end

% Get Stimuli to use
% for TT = 1:n.trials
for ii = 1:n.trials_GlobalLocal
    tmp = DATA(DATA(:,D.globallocal)==1,D.Trial);
    TT = tmp(ii);
    
    MD = DATA(TT,D.Movement_Direction); % movedir for this trial
    
    TRii = DATA(TT,D.TrainAngle_idx); % orientation for this trial
    possible_dirs = AveDir(:,:,:,TRii); % possible directions for this orientation
    
    [UTRjj, UTRii ,TRjj] = ind2sub(size(possible_dirs),find(possible_dirs == MD));
    
    DATA(TT, D.TrainAngle_movedir_idx) = TRjj;
    DATA(TT, D.AdditionalAngle_idx) = UTRii;
    DATA(TT, D.AdditionalAngle_movedir_idx) = UTRjj;
    
    % Convert indices to angles
    DATA(TT, D.AdditionalAngle)  = angles.add(DATA(TT, D.AdditionalAngle_idx),DATA(TT,D.TrainAngle_idx));
    
end

% [sum(DATA(:, D.TrainAngle_movedir) ==1) sum(DATA(:, D.TrainAngle_movedir) ==2)];
% [sum(DATA(:, D.AdditionalAngle) ==1) sum(DATA(:, D.AdditionalAngle) ==2)]
% [sum(DATA(:, D.AdditionalAngle_movedir) ==1) sum(DATA(:, D.AdditionalAngle_movedir) ==2)]

% Convert indices to angles
DATA(DATA(:,D.globallocal)==1,D.TrainAngle) = angles.train(DATA(DATA(:,D.globallocal)==1,D.TrainAngle_idx));
DATA(DATA(:,D.globallocal)==1, D.TrainAngle_movedir) = angles.movedirs(DATA(DATA(:,D.globallocal)==1, D.TrainAngle_movedir_idx));
DATA(DATA(:,D.globallocal)==1, D.AdditionalAngle_movedir) = angles.movedirs(DATA(DATA(:,D.globallocal)==1, D.AdditionalAngle_movedir_idx));


%% Setup PsychToolbox

% setup
sca
PsychDefaultSetup(2);

% Screen settings - overwrite default if only 1 screen available
monitor.screens = Screen('Screens');
if max(monitor.screens) ==0
    monitor.use = 0;
end

% Colours
colours.white = WhiteIndex(monitor.use);
colours.black = BlackIndex(monitor.use);
colours.grey = colours.white / 2;

% Keys
key.esc = 27;

% Open an on screen window
[window_Ptr, windowRect] = PsychImaging('OpenWindow', monitor.use, colours.black);

% Set up alpha-blending for smooth (anti-aliased) lines
Screen('BlendFunction', window_Ptr, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

% Get refresh rate
monitor.IFI = Screen('GetFlipInterval', window_Ptr);
monitor.refresh = round(1/monitor.IFI);

% Get the size of the on screen window
[monitor.resolution(1), monitor.resolution(2)] = Screen('WindowSize', window_Ptr);
[monitor.Center(1), monitor.Center(2)] = RectCenter(windowRect);

%% Grating Settings - convert to pixels

monitor.dva = 2*atand((0.5*monitor.width)/monitor.viewdist);
PixPerDeg = monitor.resolution(1) / monitor.dva ;    % pixels per degree

grating.spatial_freq = grating.spatial_freq_DVA/PixPerDeg; %cycles per pixel
% grating.speed_pixels = grating.speed_DVA*PixPerDeg; % pixels per second

%% Calculate Phases for movement

f.gratingMove = s.gratingMove*monitor.refresh;

% grating.Speed_cylces = grating.spatial_freq*grating.speed_pixels; % cylces per second

grating.RefreshPerCycle = monitor.refresh/grating.Speed_cylces; % monitor refreshes per cycle
grating.numcycles = f.gratingMove/grating.RefreshPerCycle;

tmp = linspace(0,2*pi*grating.numcycles, f.gratingMove+1);
grating.phase = tmp(1:end-1); % go to end - 1 to make sure phase 0°/360° isn't repeated twice in loop!
grating.phase = mod(grating.phase, 2*pi);

%% phase presentation

% reversal
for ii = 1:length(angles.movedirs)
    switch ii 
        case 1
            phase_progression{ii} =  length(grating.phase):angles.movedirs(ii):1;
        case 2
            phase_progression{ii} =  1:angles.movedirs(ii):length(grating.phase);
    end
   
end

% randomly permute starting phase
tmp = [];
for ii = 1:ceil(n.trials/length(grating.phase))
    tmp = [tmp; randperm(length(grating.phase))'];
end
DATA(:, D.starting_phase) = tmp(1:n.trials);

PHASES = NaN(2, length(grating.phase), n.trials);
%  for TT = 1:n.trials

% Phases for local motion gratings
for ii = 1:n.trials_GlobalLocal
    tmp = DATA(DATA(:,D.globallocal)==1,D.Trial);
    TT = tmp(ii);
    tmp = phase_progression{DATA(TT, D.TrainAngle_movedir_idx)}; % what the reg phase progression would be
    PHASES(1, :,TT) = [tmp(DATA(TT, D.starting_phase):end)'; tmp(1:DATA(TT, D.starting_phase)-1)'];
    
    tmp = phase_progression{DATA(TT, D.AdditionalAngle_movedir_idx)}; % what the reg phase progression would be
    PHASES(2, :,TT) = [tmp(DATA(TT, D.starting_phase):end)'; tmp(1:DATA(TT, D.starting_phase)-1)'];
end

% Phases for global motion grating - input as train angle
for ii = 1:n.trials_GlobalLocal
    tmp = DATA(DATA(:,D.globallocal)==2,D.Trial);
    TT = tmp(ii);
    tmp = phase_progression{1}; % what the reg phase progression would be
    PHASES(1, :,TT) = [tmp(DATA(TT, D.starting_phase):end)'; tmp(1:DATA(TT, D.starting_phase)-1)'];
    
end

%% Make Grating Shape

grating.diameter = monitor.resolution(1); %pixels
grating.radius = grating.diameter./2;

grating.shape = zeros(grating.diameter,grating.diameter);
try
    grating.shape = insertShape(grating.shape,'FilledCircle',[1 1 1].*grating.diameter/2,'Color','w');
catch
    warning('Run in Matlab 2014a or newer! - things will look weird if not')
    grating.shape = ones(grating.diameter,grating.diameter);
end
grating.shape = round(grating.shape(:,:,1));

%% Make Grating

GRATINGS = NaN(grating.diameter, grating.diameter,length(grating.phase));

for PH = 1:length(grating.phase)
    %% Make Sin wave
    
    x = 1:grating.diameter;
    
    y = 0.5*sin((2*pi*grating.spatial_freq).*x+grating.phase(PH))+0.5 ;
    
%     y(y>0.5) = 1;
    y = round(y).*255;
%     y = (y).*255;
    %% Make Image
    
    GRATINGS(:,:,PH) = repmat(y',1,grating.diameter).*grating.shape;
    
end

%% designate grating screen area:
grating.screenRect = [monitor.Center monitor.Center] + [-1 -1   1 1].*grating.radius;


%% Make Eliptical Mask
disp('Constructing Ellipse!')
% Settings

ellipse.radius = ellipse.diameter./2;
ellipse.radius2 = ellipse.radius-ellipse.fadeoff+5;

ellipse.npoints = 10000;
ellipse.theta = linspace(-pi, pi, ellipse.npoints);

ellipse.x = ellipse.radius(1).*cos(ellipse.theta) + monitor.Center(1);
ellipse.y = ellipse.radius(2).*sin(ellipse.theta) + monitor.Center(2);

ellipse.x2 = ellipse.radius2(1).*cos(ellipse.theta) + monitor.Center(1);
ellipse.y2 = ellipse.radius2(2).*sin(ellipse.theta) + monitor.Center(2);

% - Make faded ellipse
ellipse.IM = zeros(monitor.resolution)';

ellipse.Xuse_poly = [ellipse.x(1) ellipse.x(1) ellipse.x ellipse.x(end) ellipse.x(end)]; 
ellipse.Yuse_poly = [ellipse.y(1) ellipse.y(1) ellipse.y ellipse.y(end) ellipse.y(end)];

% Make fading
for jj = 1:ellipse.npoints
    % indices
    jjuse = jj+2;
    jj_range = (jjuse-2 ):(jjuse+2 );
    
    % Fit Line!
    ellipse.coeffs = polyfit(ellipse.Xuse_poly( jj_range), ellipse.Yuse_poly( jj_range), 1);
    
    % get coeffs
    ellipse.slope = -1/ellipse.coeffs(1);
    ellipse.CC = ellipse.Yuse_poly( jjuse) - ellipse.slope*ellipse.Xuse_poly(jjuse);
    
    % Get X and Y diff for line of length fadeoff
    II = sqrt((ellipse.fadeoff^2)/(ellipse.slope^2 +1));
%     JJ = ellipse.slope*II;
    
    % Get X, Y, and Z points for Normal to the line
    XX = linspace(ellipse.Xuse_poly(jjuse)-II,ellipse.Xuse_poly(jjuse)+II, ellipse.fadeoff);
    
%     ZZ = linspace(0,1, ellipse.fadeoff);%sigmf(1:lengthD,[0.2 50]);%
    ZZ = sigmf(1:ellipse.fadeoff,[0.2 ellipse.fadeoff/2]);%
     
    if mean(XX)>monitor.Center(1)
        ZZ=fliplr(ZZ);
    end
    
    YY = ellipse.slope*XX + ellipse.CC;
    
    XX(XX<1)=1; XX(XX>monitor.resolution(1))=monitor.resolution(1);
    YY(YY<1)=1; YY(YY>monitor.resolution(2))=monitor.resolution(2);
    
    % Fill in the points on the image
    for ii = 1:ellipse.fadeoff
        ellipse.IM(round(YY(ii)), round(XX(ii))) = ZZ(ii);
    end
    
end

% Blur edges
ellipse.idx.missed = (ellipse.IM==0);
for ii = 1:10
    tmp = mean(cat(3, [ellipse.IM(2:end,:); ellipse.IM(1,:)],[ellipse.IM(end,:); ellipse.IM(1:end-1,:)],[ellipse.IM(:,2:end) ellipse.IM(:,1)],[ellipse.IM(:,end) ellipse.IM(:,1:end-1)]),3);
    tmp2 = mean(cat(3, [tmp(2:end,:); tmp(1,:)],[tmp(end,:); tmp(1:end-1,:)],[tmp(:,2:end) tmp(:,1)],[tmp(:,end) tmp(:,1:end-1)]),3);
    ellipse.IM(ellipse.idx.missed ) =  tmp2(ellipse.idx.missed );
end

% Insert centre shape
ellipse.vectorXY = NaN(length(ellipse.x2)*2,1); ellipse.vectorXY(1:2:end-1) = ellipse.x2; ellipse.vectorXY(2:2:end) = ellipse.y2;
ellipse.IM2 = zeros(monitor.resolution)';
ellipse.IM2 = insertShape(ellipse.IM2,'FilledPolygon',ellipse.vectorXY', 'Color', 'white');
ellipse.IM2 = round(ellipse.IM2(:,:,1));

% Put it all together
ellipse.IM3 = imgaussfilt(ellipse.IM,2) + ellipse.IM2;
ellipse.IM3(ellipse.IM3>1)=1;
ellipse.IM3(ellipse.IM3<0)=0;

% figure; imshow(ellipse.IM3)
% figure; imshow(abs(ellipse.IM3-1))

disp('Done Constructing Ellipse!')

%% Make Textures
gratingTextures = NaN(length(grating.phase),1);

for PH = 1:length(grating.phase)
    
    imtmp = GRATINGS(:,:,PH);%.*grating.Gauss;
    theImage = repmat(uint8(imtmp),1,1,4);
    theImage(:,:,4) = imtmp./2;%change black to transparent
    
    % Make the image into a texture
    gratingTextures(PH) = Screen('MakeTexture', window_Ptr,theImage);
    
end

% Ellipse Texture

Ellipse.mask = zeros(monitor.resolution(2), monitor.resolution(1),4);
Ellipse.mask(:,:,4) = abs(ellipse.IM3-1);
MASK = Screen('MakeTexture', window_Ptr,Ellipse.mask);

%% Preallocate

fliptime = NaN(f.gratingMove, n.trials);

%% Start Presentation!

for TRIAL = 1:n.trials
    disp('--------------')
    disp(['Trial: ' num2str(DATA(TRIAL,D.Trial)) ' of ' num2str(n.trials)])
    disp(['Ave Dir: ' num2str(DATA(TRIAL,D.Movement_Direction))])
    disp(['Train/UnTrain Orent: ' num2str(DATA(TRIAL,D.TrainAngle))])
    disp(['Additional Orient: ' num2str(DATA(TRIAL,D.AdditionalAngle))])
    
    for FRAME = 1:f.gratingMove
        
        switch DATA(TRIAL,D.globallocal)
            case 1
                % Train grating
                Screen('DrawTextures', window_Ptr,...
                    gratingTextures(PHASES(1, FRAME,TRIAL)), ... % phase
                    [], grating.screenRect, ...
                    DATA(TRIAL,D.TrainAngle)) % angle
                
                % Additional grating
                Screen('DrawTextures', window_Ptr,...
                    gratingTextures(PHASES(2, FRAME,TRIAL)), ... % phase
                    [], grating.screenRect, ...
                    DATA(TRIAL,D.AdditionalAngle)) % angle
                
            case 2
                Screen('DrawTextures', window_Ptr,...
                    gratingTextures(PHASES(1, FRAME,TRIAL)), ... % phase
                    [], grating.screenRect, ...
                    DATA(TRIAL,D.Movement_Direction)-90) % angle
        end
        % Mask
        Screen('DrawTextures', window_Ptr,...
            MASK, ... % phase
            [], windowRect) % angle
        
        % Flip to the screen
        fliptime(FRAME, TRIAL) = Screen('Flip', window_Ptr);
    end
    
    [~, ~, keyCode, ~] = KbCheck();
    
    if find(keyCode)==key.esc
        sca
        break;
    end
end

%% Clean up
sca

%% Save!
save([direct.Data fname.session '.mat'])

%% Flip timing

figure;
plot(diff(fliptime))
% 26.66 mins