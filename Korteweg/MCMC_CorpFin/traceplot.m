function traceplot(x, G, true_x)
% Copyright (C) 2011 by Arthur Korteweg
%
% graphs a trace plot of the first G draws of x and draws a red line at the 
% true value (optional, if true_x specified).

hold on;
plot(x); 
v = axis; axis([1 G+1 v(3:4)]); set(gca,'XTick',0:G/5:G); xlabel('Iteration')
if nargin > 2
    line([1 G+1], [true_x true_x], 'Color', 'r', 'LineWidth', 2);
    v = axis; axis([1 G+1 min(v(3),true_x)-.01 max(v(4),true_x)+.01]);
end
box;
hold off;
