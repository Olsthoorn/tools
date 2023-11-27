% Convert boring database into a real database with key {putcode filter}
% Usage of database boring.mat

F = '~/GRWMODELS/AWD/AWD-data/Boringen/boring.mat';
load(F)


obs = {
        '24H189'
        '24H304'
        '24H190'
        '24H425'
        '24H717'
        '24H070'
        '24H018'
        '24H428'
        '25C008'
        '25C332'
        '25C012'};


bore = boringObj(obs,boring);

bore.display;

figure; hold on;
bore.plot;
