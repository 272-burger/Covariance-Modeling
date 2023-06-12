%% comparing all power curves
powerCurveComparison = [powerCurve_sk',powerCurve_loc,powerCurve_cceGstarB,...
    powerCurveGstar_im,...
    powerCurveGstar_crs'];

% SK
p = plot(altPowerCurve,powerCurveComparison(:,1),'-s',...
    'MarkerIndices',1:30:length(altPowerCurve),...
    'MarkerSize',10);
p.LineWidth = 2;
hold on

% UNIT
p = plot(altPowerCurve,powerCurveComparison(:,2),'-+',...
    'MarkerIndices',1:30:length(altPowerCurve),...
    'MarkerSize',10);
p.LineWidth = 2;

% CCE
p = plot(altPowerCurve,powerCurveComparison(:,3),'-x',...
    'MarkerIndices',1:30:length(altPowerCurve),...
    'MarkerSize',10);
p.LineWidth = 2;

% IM
p = plot(altPowerCurve,powerCurveComparison(:,4),'-o',...
    'MarkerIndices',1:30:length(altPowerCurve),...
    'MarkerSize',10);
p.LineWidth = 2;

% CRS
p = plot(altPowerCurve,powerCurveComparison(:,5),'-^',...
    'MarkerIndices',1:30:length(altPowerCurve),...
    'MarkerSize',10);
p.LineWidth = 2;
hold off

lgd = legend('SK','UNIT','CCE','IM','CRS','location','SouthEast');
ylabel('Empirical Rejection Rate');
if lower(modelName) == "ols"
    if lower(sampleScale) == "small"
        xlim([-3,3])
    else
        xlim([-2,2])
    end
else
    if lower(sampleScale) == "small"
        xlim([-2,2])
    else
        xlim([-1,1])
    end
end
set(gca,'FontSize',18,'FontName','Times New Roman')
xlabel('\theta^{alt}');
x0=10;
y0=10;
width=400;
height=400;
set(gcf,'position',[x0,y0,width,height]);
figruePath = ['../output/power_curve_',modelName,'_',spatialModel,...
    '_',sampleScale,'.pdf'];
saveas(gcf,figruePath)
close(gcf)