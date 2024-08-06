clear all
load SimulStock2

YearRange =2001:2010;

Year = reshape(repmat(YearRange,4,1),4*length(YearRange),1);
Year = [Year(2:end);YearRange(end)+1];

Q = repmat([2;3;4;1],length(YearRange),1);
N = length(Year);

dateVector = datestr([Year,3*(Q-1)+1,ones(N,1),zeros(N,1),zeros(N,1),zeros(N,1)]);

dateSeries = datenum(dateVector)-1;


f1 = figure(1);
plot(dateSeries, baa(:,1), '-k', dateSeries, baa(:,2), ':g', dateSeries, baa(:,3), '-.r', dateSeries, baa(:,4), '--b', 'LineWidth',1.5)
grid on;
datetick('x',17)
axis([min(dateSeries) max(dateSeries) -0.5 2])
legend('Estimated g(t)', 'C-EW','CRSP-EW','CRSP-VW', 'Location', 'northwest')
legend('boxoff')
print (f1, 'FigR2Left','-dpng')
%%

f2 = figure(2);
startDate=size(baa,1)-size(baa_cut,1)+1;
plot(dateSeries(startDate:end), baa_cut(:,1), '-k', dateSeries(startDate:end), baa_cut(:,2), ':g', dateSeries(startDate:end), baa_cut(:,3), '-.r', dateSeries(startDate:end), baa_cut(:,4), '--b', 'LineWidth',1.5)
grid on;
datetick('x',17)
axis([min(dateSeries(startDate:end)) max(dateSeries(5:end)) -0.2 1.25])
legend('Estimated g(t)', 'C-EW','CRSP-EW','CRSP-VW', 'Location', 'northwest')
legend('boxoff')
print (f2, 'FigR2Right','-dpng')

%%

clear all

load Vint2000

YearRange =2000:2012;
Year = reshape(repmat(YearRange,4,1),4*length(YearRange),1);
Year = [Year(2:end);YearRange(end)+1];

Q = repmat([2;3;4;1],length(YearRange),1);
N = length(Year);

dateVector = datestr([Year,3*(Q-1)+1,ones(N,1),zeros(N,1),zeros(N,1),zeros(N,1)]);

dateSeries = datenum(dateVector)-1;

f3 = figure(3);
plot(dateSeries, aaa(:,1), '-b', dateSeries, aaa(:,2), '-g', 'LineWidth',1.5)
grid on;
datetick('x',17)
axis([min(dateSeries) max(dateSeries) -2 1.5])
legend('VC returns', 'BO returns','Location','northwest')
legend('boxoff')
print (f1, 'FigR3','-dpng')

