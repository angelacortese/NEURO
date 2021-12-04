clc 
clear all 
close all 

%% LOAD DATA
load 0kg_signals.mat;
data0 = signal.emgdata;
emgtime0 = signal.emgtime;
pos0 = signal.posdata;
postime0 = signal.postime;

load 1kg_signals.mat;
data1 = signal.emgdata;
emgtime1 = signal.emgtime;
pos1 = signal.posdata;
postime1 = signal.postime;

load 2kg_signals.mat;
data2 = signal.emgdata;
emgtime2 = signal.emgtime;
pos2 = signal.posdata;
postime2 = signal.postime;

load 3kg_signals.mat;
data3 = signal.emgdata;
emgtime3 = signal.emgtime;
pos3 = signal.posdata;
postime3 = signal.postime;

%% DETREND AND PLOT

data0 = detrend(data0);
data1 = detrend(data1);
data2 = detrend(data2);
data3 = detrend(data3);

pos0 = detrend(pos0);
pos1 = detrend(pos1);
pos2 = detrend(pos2);
pos3 = detrend(pos3);

subplot(2,2,1)
plot(emgtime0,data0,'b');
hold on;
plot(postime0,pos0,'r');
title('0KG');

subplot(2,2,2)
plot(emgtime1,data1,'b');
hold on;
plot(postime1,pos1,'r');
title('1KG');

subplot(2,2,3)
plot(emgtime2,data2,'b');
hold on;
plot(postime2,pos2,'r');
title('2KG');

subplot(2,2,4)
plot(emgtime3,data3,'b');
hold on;
plot(postime3,pos3,'r');
title('3KG');

%% SPECTRUM ANALYSIS
fs = 1024;
figure();

% 0KG
N = length(data0);
f = fs*(0:(N/2))/N;
DFT_data = fft(data0);
pl = abs(DFT_data/N);
pl = pl(1:N/2+1);
pl(2:end-1) = 2*pl(2:end-1);
subplot(2,2,1);
plot(f,pl);

% 1KG
N = length(data1);
f = fs*(0:(N/2))/N;
DFT_data = fft(data1);
pl = abs(DFT_data/N);
pl = pl(1:N/2+1);
pl(2:end-1) = 2*pl(2:end-1);
subplot(2,2,2);
plot(f,pl);

% 0KG
N = length(data2);
f = fs*(0:(N/2))/N;
DFT_data = fft(data2);
pl = abs(DFT_data/N);
pl = pl(1:N/2+1);
pl(2:end-1) = 2*pl(2:end-1);
subplot(2,2,3);
plot(f,pl);

% 0KG
N = length(data3);
f = fs*(0:(N/2))/N;
DFT_data = fft(data3);
pl = abs(DFT_data/N);
pl = pl(1:N/2+1);
pl(2:end-1) = 2*pl(2:end-1);
subplot(2,2,4);
plot(f,pl);

%% FILTERING
figure();
fcuthigh = 10; %Hz
fcutlow = 200; %Hz
[B,A]= butter(4,[fcuthigh fcutlow]/512, 'bandpass');

Fnotch = [50 150 250]/(fs/2);  % Notch Frequency
BW     = 5/(fs/2); % Bandwidth
Apass  = 1;    % Bandwidth Attenuation
mycomb = zeros(2,6);

for i=1:length(Fnotch)
    [b, a] = iirnotch(Fnotch(i), BW, Apass);
    mycomb(i,:) = [b,a];
end

filtered0 = filtfilt(B,A,data0);
dataOut0 = sosfilt(mycomb,filtered0);
subplot(2,2,1);
plot(emgtime0,dataOut0);

filtered1 = filtfilt(B,A,data1);
dataOut1 = sosfilt(mycomb,filtered1);
subplot(2,2,2);
plot(emgtime1,dataOut1);

filtered2 = filtfilt(B,A,data2);
dataOut2 = sosfilt(mycomb,filtered2);
subplot(2,2,3);
plot(emgtime2,dataOut2);

filtered3 = filtfilt(B,A,data3);
dataOut3 = sosfilt(mycomb,filtered3);
subplot(2,2,4);
plot(emgtime3,dataOut3);

%% RECIFICATION AND ENVELOPE
window = 500;

rec_signal0 = abs(dataOut0);
envelope0 = sqrt(movmean((rec_signal0.^2),window));

rec_signal1 = abs(dataOut1);
envelope1 = sqrt(movmean((rec_signal1.^2),window));

rec_signal2 = abs(dataOut2);
envelope2 = sqrt(movmean((rec_signal2.^2),window));

rec_signal3 = abs(dataOut3);
envelope3 = sqrt(movmean((rec_signal3.^2),window));

%% PLOT OF RESULTS
figure();

subplot(2,2,1);
plot(emgtime0,rec_signal0);
hold on;
plot(emgtime0,envelope0,'-r');

subplot(2,2,2);
plot(emgtime1,rec_signal1);
hold on;
plot(emgtime1,envelope1,'-r');

subplot(2,2,3);
plot(emgtime2,rec_signal2);
hold on;
plot(emgtime2,envelope2,'-r');

subplot(2,2,4);
plot(emgtime3,rec_signal3);
hold on;
plot(emgtime3,envelope3,'-r');


% 
% figure,
% pspectrum(data0,fs);
% [B,A] = butter(2, 0.0195,'low'); % Wn = 10/512 cutoff 10hz
% freqz(B,A);
% dataOut = filter(B,A,data0);
% figure,
% plot(signal.emgtime,dataOut);
% 





