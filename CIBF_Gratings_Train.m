clear
clc
close all

%% Session Settings

subject_ID = 'M0'; %To keep track of the subject across sessions
angles.train_idx = 1; %1 = -45°, 2 = 45°!

subject_SesionNotes = 'Train'; % Part of saved name, can add things like Day1, Day2 etc.

%% Screen Settings

monitor.use = 1;
monitor.resolution = [1920 1080];

monitor.width = 530;%[530 345]; % mm
monitor.viewdist = 250; % mm

%% Grating Settings

grating.spatial_freq_DVA = 0.04;%; %cycles per pixel
grating.Speed_cylces = 1; % Hz (cycles per second)

ellipse.fadeoff = 50;
ellipse.diameter = [1820 980];

%% Timing settings

s.gratingMove = 1; %seconds
key.enter = 13;
key.esc = 27;

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


%% Orientation settings

angles.train = [-45 45]; % the two possible training angles;
Orient.all = [-75   -45   -15    15    45    75];

Orient.rewarded = angles.train(angles.train_idx);

Orient.idxAll.rewarded = find(Orient.all==Orient.rewarded);
Orient.idxAll.neutral = find(Orient.all~=Orient.rewarded);

Orient.neutral = Orient.all(Orient.idxAll.neutral);

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


%% Grating Settings  - convert to pixels
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

%% Counterbalancing - setup DATA

n.phases = length(unique(grating.phase));
n.orients = length(Orient.all);
n.NeutOrients = length(Orient.neutral);
n.trials = 2*(n.NeutOrients)*n.phases;


% DATA
str.DATA = {
    'Trial'
    'Block'
    
    'Train_Orient'
    
    'RewVsNeut'
    
    'Orient'
    'OrientAll_idx'
    
    'Phase'};

n.D = length(str.DATA);

for ii = 1:n.D
   D.(str.DATA{ii}) = ii; 
end

DATA = NaN(n.trials, n.D);

% Trials
DATA(:,D.Trial) = 1:n.trials;

% BLOCK
DATA(:,D.Block) = ones(n.trials, 1).*1;

% Train Orient
DATA(:,D.Train_Orient) = ones(n.trials, 1).*Orient.rewarded;

% RewVsNeut
tmp = [];
for ii = 1:2
    tmp = [tmp; ones(n.trials/2,1)*ii];
end

tmp = tmp(randperm(n.trials));
DATA(:,D.RewVsNeut) =tmp;

% Reward Trials Orient
DATA(DATA(:,D.RewVsNeut)==1,D.Orient) = Orient.rewarded;
DATA(DATA(:,D.RewVsNeut)==1,D.OrientAll_idx) = Orient.idxAll.rewarded;

% Neutral Trials Orient
tmp = [];
for ii = 1:n.NeutOrients
    tmp = [tmp; ones(n.phases,1).*ii];
end
tmp = tmp(randperm(n.trials/2));

DATA(DATA(:,D.RewVsNeut)==2,D.Orient) = Orient.neutral(tmp);
DATA(DATA(:,D.RewVsNeut)==2,D.OrientAll_idx) = Orient.idxAll.neutral(tmp);

% Reward Trials phase
idx.tmp = randperm(n.phases*n.NeutOrients);
tmp = repmat(1:n.phases, 1, n.NeutOrients);
DATA(DATA(:,D.OrientAll_idx)==Orient.idxAll.rewarded,D.Phase) = tmp(idx.tmp);

% Neutral Trials phase
for ii = Orient.idxAll.neutral
    tmp = randperm(n.phases);
    DATA(DATA(:,D.OrientAll_idx)==ii,D.Phase) = tmp;
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

DATA_Blank = DATA; 

BLOCK = 1;
TRIAL = 0;
TrialUse = 0;

breaker = 0;

str.reward = {'Reward' 'Neutral'};

%% Start Presentation!

while true
    TRIAL = TRIAL + 1;
    TrialUse = TrialUse + 1;
    disp('--------------')
    disp(['Trial: ' num2str(DATA(TRIAL,D.Trial)) ' of ' num2str(n.trials)])
    disp(['Orientation : ' num2str(DATA(TRIAL,D.Orient)) '° (' str.reward{DATA(TRIAL,D.RewVsNeut)} ')' ])
    
    % display blank screen
    for frame = 1:round(monitor.refresh*0.2)
        fliptime = Screen('Flip', window_Ptr);
    end
    
%     display grating
    while true

        % Train grating
        Screen('DrawTextures', window_Ptr,...
            gratingTextures(DATA(TRIAL, D.Phase)), ... % phase
            [], grating.screenRect, ...
            DATA(TRIAL,D.Orient)) % angle
        
         % Mask
        Screen('DrawTextures', window_Ptr,...
            MASK, ... % phase
            [], windowRect) % angle
        
        % Flip to the screen
        fliptime = Screen('Flip', window_Ptr);
        
        % Change over if key press

        [~, ~, keyCode, ~] = KbCheck();
        if find(keyCode)==key.enter
            break;
        elseif  find(keyCode)==key.esc
            breaker = 1;
            break;
        end

    end
    
    %% Break
    
    if  breaker == 1
       break; 
    end
    %% Move on
    
    if TrialUse == n.trials
        BLOCK = BLOCK+1;
        TrialUse = 0;
        
        tmp = DATA_Blank;
        tmp(:,D.Block) =  ones(n.trials, 1).*BLOCK;
        tmp(:,D.Trial) = (1:n.trials) + TRIAL;
        
        DATA = [DATA; tmp];
    end
    
end

%% Clean up
sca

%% Save!
save([direct.Data fname.session '.mat'])