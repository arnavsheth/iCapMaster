function histplot(x, N, true_x)
% Plots a histogram of x with N bins, fits a kernel density, and draws a
% red line at the true value (optional, if true_x specified).

hold on;
hist(x, N);
h = findobj(gca,'Type','patch');
set(h,'FaceColor','none','EdgeColor','black')
[f,z] = ksdensity(x);
plot(z, f*max(hist(x, N))/max(f), 'Color', 'b', 'LineWidth', 0.5);
if nargin > 2 
    v = axis;
    line([true_x true_x], [0 (v(4)+max(hist(x, N)))/2], 'Color', 'r', 'LineWidth', 2);
end
box;
hold off;
