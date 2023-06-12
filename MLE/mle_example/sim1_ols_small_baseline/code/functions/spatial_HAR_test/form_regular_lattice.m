% This program converts irregulat lattice into  regular on

 
function [jx,jy]= form_regular_lattice(x,y)

% This program converts irregulat lattice into  regular lattice

xs = sort(x);
tempx = abs(diff(xs));
tempx(tempx==0)=[];
dx = median(tempx);

ys = sort(y);
tempy = abs(diff(ys));
tempy(tempy==0)=[];
dy = median(tempy);


%% Normalization
 x = (x-min(x))/dx;
 y = (y-min(y))/dy;

%% Minimum distance
xy =[x y];
d=inf;
for i=1:1:length(x);
    for j=i+1:1:length(x)
        temp = max(abs(xy(i,:)-xy(j,:)));
        if temp == 0
            temp = inf;
        end;
        d = min(d,temp);
    end
end;

jx = ceil((x-min(x))/d);
jy = ceil((y-min(y))/d);
